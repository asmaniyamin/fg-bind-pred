# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Trainer for the counterfactual FG-importance model (model2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from ..data.label_builder import build_targets_from_batch
from .metrics import (
    compute_per_task_metrics,
    compute_prediction_diagnostics,
    per_chemotype_metrics,
)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        config,
        device: str,
        baseline_means: dict[str, float] | None = None,
        outlier_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.baseline_means = dict(baseline_means or {})
        # Per-task (median - k*MAD, median + k*MAD) bounds derived once from the train set.
        # Applied inside build_targets_from_batch to mask out major outliers as invalid targets.
        self.outlier_bounds = dict(outlier_bounds or {})
        self.model.to(device)
        self._freeze_active = False

    def _build_targets(self, batch):
        return build_targets_from_batch(
            batch,
            self.model.config.cross_attn_cutoff,
            max_abs_importance=float(getattr(self.config, "max_abs_importance", 0.0) or 0.0),
            hotspot_top_k=int(self.model.config.hotspot_top_k),
            hotspot_min_abs=float(self.model.config.hotspot_min_abs),
            outlier_bounds=self.outlier_bounds or None,
            include_contact_aux=bool(getattr(self.model.config, "use_contact_aux_head", False)),
        )

    def _get_metric_value(self, payload: dict[str, Any], metric_path: str) -> float:
        value: Any = payload
        for key in metric_path.split("."):
            if not isinstance(value, dict) or key not in value:
                return float("nan")
            value = value[key]
        if isinstance(value, (int, float)):
            return float(value)
        return float("nan")

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        task_sums: dict[str, float] = {}
        accum_steps = max(int(getattr(self.config, "gradient_accumulation_steps", 1)), 1)
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loader):
            batch = self._move_to_device(batch)
            targets, masks = self._build_targets(batch)
            preds = self.model(batch)
            loss, per_task = self.loss_fn(preds, targets, masks)
            loss_for_backward = loss / accum_steps
            loss_for_backward.backward()
            should_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(loader))
            if should_step:
                clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and getattr(self.config, "scheduler_step_on", "batch") == "batch":
                    self.scheduler.step()

            running_loss += float(loss.item())
            n_batches += 1
            for key, value in per_task.items():
                task_sums[key] = task_sums.get(key, 0.0) + float(value.item())
            print_every = int(getattr(self.config, "print_batch_every", 0))
            if print_every > 0 and ((batch_idx + 1) % print_every == 0 or (batch_idx + 1 == len(loader))):
                print(
                    f"  batch {batch_idx + 1}/{len(loader)} loss={float(loss.item()):.6f}",
                    flush=True,
                )

        metrics = {"loss": running_loss / max(n_batches, 1), "lr": self.optimizer.param_groups[0]["lr"]}
        for key, value in task_sums.items():
            metrics[f"loss_{key}"] = value / max(n_batches, 1)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader, split: str = "val") -> dict[str, Any]:
        del split
        self.model.eval()
        all_preds, all_targets, all_masks = self._collect_all_predictions(loader)
        if not all_preds:
            return {}
        metrics = compute_per_task_metrics(all_preds, all_targets, all_masks)
        metrics["per_chemotype"] = per_chemotype_metrics(
            all_preds["importance_ghost"],
            all_targets["importance_ghost"],
            all_targets["fg_type"],
            all_masks.get("importance_ghost"),
        )
        metrics["diagnostics"] = compute_prediction_diagnostics(
            all_preds,
            all_targets,
            all_masks,
            baseline_means=self.baseline_means,
        )
        return metrics

    def train(self, train_loader, val_loader, num_epochs: int, checkpoint_dir: str | Path | None = None):
        early_mode = getattr(self.config, "early_stopping_mode", "max")
        best_score = float("-inf") if early_mode == "max" else float("inf")
        patience = 0
        history = []
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(num_epochs):
            self._apply_freeze_policy(epoch)
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, split="val") if len(val_loader) else {}
            val_loss_metrics = self.evaluate_losses(val_loader) if len(val_loader) else {}
            if val_loss_metrics:
                val_metrics = dict(val_metrics)
                val_metrics.update(val_loss_metrics)
            if self.scheduler is not None and getattr(self.config, "scheduler_step_on", "batch") == "epoch":
                if getattr(self.config, "scheduler_type", "") == "plateau":
                    self.scheduler.step(val_metrics.get("loss", float("nan")))
                else:
                    self.scheduler.step()
            metric_path = getattr(self.config, "early_stopping_metric", "importance_ghost.per_complex_spearman_r")
            score = self._get_metric_value(val_metrics, metric_path)
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            self._print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics)
            if checkpoint_dir is not None:
                self.save_checkpoint(str(checkpoint_dir / "last.pt"), epoch, {"train": train_metrics, "val": val_metrics})
            improved = False
            if score == score:
                if early_mode == "max":
                    improved = score > best_score
                else:
                    improved = score < best_score
            if improved:
                best_score = score
                patience = 0
                if checkpoint_dir is not None:
                    self.save_checkpoint(str(checkpoint_dir / "best.pt"), epoch, {"train": train_metrics, "val": val_metrics})
            else:
                if checkpoint_dir is not None and epoch == 0 and not (checkpoint_dir / "best.pt").exists():
                    self.save_checkpoint(str(checkpoint_dir / "best.pt"), epoch, {"train": train_metrics, "val": val_metrics})
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    break
        return history

    def _apply_freeze_policy(self, epoch: int) -> None:
        freeze_modules = tuple(getattr(self.config, "freeze_modules", ()))
        freeze_epochs = int(getattr(self.config, "freeze_epochs", 0))
        if not freeze_modules or freeze_epochs <= 0:
            return
        if epoch < freeze_epochs and not self._freeze_active:
            self._set_modules_trainable(freeze_modules, trainable=False)
            self._freeze_active = True
            print(f"freeze active modules={','.join(freeze_modules)} until epoch {freeze_epochs}", flush=True)
        elif epoch >= freeze_epochs and self._freeze_active:
            self._set_modules_trainable(freeze_modules, trainable=True)
            self._freeze_active = False
            print(f"freeze released modules={','.join(freeze_modules)}", flush=True)

    def _set_modules_trainable(self, module_names: tuple[str, ...], trainable: bool) -> None:
        if not hasattr(self.model, "module_groups"):
            return
        groups = self.model.module_groups()
        for module_name in module_names:
            for module in groups.get(module_name, []):
                for param in module.parameters():
                    param.requires_grad = trainable

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "loss_state": self.loss_fn.state_dict(),
            "metrics": metrics,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> tuple[int, dict]:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler is not None and payload.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.loss_fn.load_state_dict(payload["loss_state"])
        return int(payload["epoch"]), dict(payload.get("metrics", {}))

    def _move_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self._move_to_device(value) for value in batch]
        if isinstance(batch, tuple):
            return tuple(self._move_to_device(value) for value in batch)
        return batch

    @torch.no_grad()
    def _collect_all_predictions(self, loader) -> tuple[dict, dict, dict]:
        pred_keys = ("importance", "importance_ghost", "importance_charge", "importance_lj", "hotspot")
        target_keys = (
            "importance",
            "importance_ghost",
            "importance_charge",
            "importance_lj",
            "abs_importance",
            "hotspot",
            "fg_batch",
            "fg_type",
            "delta_total",
        )
        mask_keys = ("fg", "importance_ghost", "importance_charge", "importance_lj", "hotspot", "delta_total")
        pred_parts: dict[str, list[torch.Tensor]] = {key: [] for key in pred_keys}
        target_parts: dict[str, list[torch.Tensor]] = {key: [] for key in target_keys}
        mask_parts: dict[str, list[torch.Tensor]] = {key: [] for key in mask_keys}

        for batch in loader:
            batch = self._move_to_device(batch)
            targets, masks = self._build_targets(batch)
            preds = self.model(batch)

            for key in pred_keys:
                pred_parts[key].append(preds[key].detach().cpu())
            for key in target_keys:
                target_parts[key].append(targets[key].detach().cpu())
            for key in mask_keys:
                if key in masks:
                    mask_parts[key].append(masks[key].detach().cpu())

        if not pred_parts["importance_ghost"]:
            return {}, {}, {}
        all_preds = {key: torch.cat(values, dim=0) for key, values in pred_parts.items()}
        all_targets = {key: torch.cat(values, dim=0) for key, values in target_parts.items()}
        all_masks = {key: torch.cat(values, dim=0) for key, values in mask_parts.items() if values}
        return all_preds, all_targets, all_masks

    @torch.no_grad()
    def evaluate_losses(self, loader) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        task_sums: dict[str, float] = {}
        for batch in loader:
            batch = self._move_to_device(batch)
            targets, masks = self._build_targets(batch)
            preds = self.model(batch)
            loss, per_task = self.loss_fn(preds, targets, masks)
            running_loss += float(loss.item())
            n_batches += 1
            for key, value in per_task.items():
                task_sums[key] = task_sums.get(key, 0.0) + float(value.item())
        if n_batches == 0:
            return {}
        metrics = {"loss": running_loss / n_batches}
        for key, value in task_sums.items():
            metrics[f"loss_{key}"] = value / n_batches
        return metrics

    def _print_epoch_summary(self, epoch: int, num_epochs: int, train_metrics: dict[str, float], val_metrics: dict[str, Any]) -> None:
        train_loss = float(train_metrics.get("loss", float("nan")))
        val_loss = float(val_metrics.get("loss", float("nan"))) if val_metrics else float("nan")
        ghost = val_metrics.get("importance_ghost", {}) if val_metrics else {}
        val_ghost_spearman = ghost.get("per_complex_spearman_r", float("nan"))
        val_ghost_pearson = ghost.get("pearson_r", float("nan"))
        val_top1 = ghost.get("top1_hit_rate", float("nan"))
        val_hotspot_auc = val_metrics.get("hotspot", {}).get("roc_auc", float("nan")) if val_metrics else float("nan")
        monitor_path = getattr(self.config, "early_stopping_metric", "importance_ghost.per_complex_spearman_r")
        monitor_value = self._get_metric_value(val_metrics, monitor_path) if val_metrics else float("nan")
        print(
            f"epoch {epoch + 1}/{num_epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_ghost_pcc_spearman={val_ghost_spearman:.6f} "
            f"val_ghost_pearson={val_ghost_pearson:.6f} "
            f"val_top1_hit={val_top1:.6f} "
            f"val_hotspot_auc={val_hotspot_auc:.6f} "
            f"monitor[{monitor_path}]={monitor_value:.6f}",
            flush=True,
        )

    @torch.no_grad()
    def collect_prediction_rows(self, loader) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.model.eval()
        fg_rows: list[dict[str, Any]] = []
        complex_rows: list[dict[str, Any]] = []
        for batch in loader:
            batch = self._move_to_device(batch)
            targets, masks = self._build_targets(batch)
            preds = self.model(batch)

            metadata = batch["metadata"]
            fg_ptr = batch["functional_groups"]["fg_ptr"].detach().cpu()
            fg_type = batch["functional_groups"]["fg_type"].detach().cpu()
            fg_parent_type = batch["functional_groups"]["fg_parent_type"].detach().cpu()

            importance_pred = preds["importance"].detach().cpu()
            hotspot_pred = preds["hotspot"].detach().cpu()
            importance_target = targets["importance"].detach().cpu()
            hotspot_target = targets["hotspot"].detach().cpu()
            abs_target = targets["abs_importance"].detach().cpu()
            delta_total_target = targets.get("delta_total")
            if delta_total_target is not None:
                delta_total_target = delta_total_target.detach().cpu()
            delta_total_mask = masks.get("delta_total")
            if delta_total_mask is not None:
                delta_total_mask = delta_total_mask.detach().cpu()

            ghost_mask = masks["importance_ghost"].detach().cpu()
            charge_mask = masks["importance_charge"].detach().cpu()
            lj_mask = masks["importance_lj"].detach().cpu()

            for complex_idx, pdb_id in enumerate(metadata["pdb_ids"]):
                fg_lo = int(fg_ptr[complex_idx].item())
                fg_hi = int(fg_ptr[complex_idx + 1].item())
                ligand_id = metadata["ligand_ids"][complex_idx]
                smiles = metadata["smiles"][complex_idx]

                n_valid = int(ghost_mask[fg_lo:fg_hi].sum().item())
                row = {
                    "pdb_id": pdb_id,
                    "ligand_id": ligand_id,
                    "smiles": smiles,
                    "n_fg": fg_hi - fg_lo,
                    "n_fg_valid_ghost": n_valid,
                    "pred_ghost_sum": float(importance_pred[fg_lo:fg_hi, 0].sum().item()),
                    "target_ghost_sum": float(importance_target[fg_lo:fg_hi, 0].sum().item()),
                    "target_max_abs_ghost": float(abs_target[fg_lo:fg_hi].max().item()) if fg_hi > fg_lo else float("nan"),
                    "pred_max_abs_ghost": float(importance_pred[fg_lo:fg_hi, 0].abs().max().item()) if fg_hi > fg_lo else float("nan"),
                }
                if delta_total_target is not None:
                    row["target_baseline_delta_total"] = float(delta_total_target[complex_idx].item())
                    if delta_total_mask is not None:
                        row["mask_baseline_delta_total"] = bool(delta_total_mask[complex_idx].item())
                complex_rows.append(row)

                for local_fg_idx, global_fg_idx in enumerate(range(fg_lo, fg_hi)):
                    fg_rows.append(
                        {
                            "pdb_id": pdb_id,
                            "ligand_id": ligand_id,
                            "smiles": smiles,
                            "fg_index_in_complex": local_fg_idx,
                            "fg_global_index": global_fg_idx,
                            "fg_type_id": int(fg_type[global_fg_idx].item()),
                            "fg_parent_type_id": int(fg_parent_type[global_fg_idx].item()),
                            "pred_importance_ghost": float(importance_pred[global_fg_idx, 0].item()),
                            "pred_importance_charge": float(importance_pred[global_fg_idx, 1].item()),
                            "pred_importance_lj": float(importance_pred[global_fg_idx, 2].item()),
                            "pred_hotspot_logit": float(hotspot_pred[global_fg_idx].item()),
                            "pred_hotspot_prob": float(torch.sigmoid(hotspot_pred[global_fg_idx]).item()),
                            "target_importance_ghost": float(importance_target[global_fg_idx, 0].item()),
                            "target_importance_charge": float(importance_target[global_fg_idx, 1].item()),
                            "target_importance_lj": float(importance_target[global_fg_idx, 2].item()),
                            "target_abs_importance_ghost": float(abs_target[global_fg_idx].item()),
                            "target_hotspot": float(hotspot_target[global_fg_idx].item()),
                            "mask_importance_ghost": bool(ghost_mask[global_fg_idx].item()),
                            "mask_importance_charge": bool(charge_mask[global_fg_idx].item()),
                            "mask_importance_lj": bool(lj_mask[global_fg_idx].item()),
                        }
                    )
        return fg_rows, complex_rows
