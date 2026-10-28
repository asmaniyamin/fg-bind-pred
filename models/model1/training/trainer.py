# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from ..data.label_builder import build_targets_from_batch
from .metrics import compute_per_task_metrics, compute_prediction_diagnostics, fg_rank_correlation, per_chemotype_metrics


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
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.baseline_means = dict(baseline_means or {})
        self.model.to(device)
        self._freeze_active = False

    def _build_targets(self, batch):
        return build_targets_from_batch(
            batch,
            self.model.config.cross_attn_cutoff,
            max_abs_delta_total=float(getattr(self.config, "max_abs_delta_total", 0.0) or 0.0),
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
        contact_total_pred = all_preds.get("contact_total", all_preds["contact"].sum(dim=-1))
        contact_total_target = all_targets.get("contact_total", all_targets["contact"].sum(dim=-1))
        contact_total_mask = all_masks.get("contact_total", all_masks.get("contact", all_masks.get("fg")))
        metrics["fg_rank_correlation"] = fg_rank_correlation(
            contact_total_pred,
            contact_total_target,
            all_targets["fg_batch"],
            contact_total_mask,
        )
        metrics["per_chemotype_metrics"] = per_chemotype_metrics(
            contact_total_pred,
            contact_total_target,
            all_targets["fg_type"],
            contact_total_mask,
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
            metric_path = getattr(self.config, "early_stopping_metric", "contact_total.pearson_r")
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
        pred_parts: dict[str, list[torch.Tensor]] = {"contact": [], "contact_total": [], "preorg": [], "delta_total": []}
        target_parts: dict[str, list[torch.Tensor]] = {
            "contact": [],
            "contact_total": [],
            "preorg": [],
            "delta_total": [],
            "fg_batch": [],
            "fg_type": [],
        }
        mask_parts: dict[str, list[torch.Tensor]] = {
            "fg": [],
            "contact": [],
            "contact_total": [],
            "preorg": [],
            "delta_total": [],
        }

        for batch in loader:
            batch = self._move_to_device(batch)
            targets, masks = self._build_targets(batch)
            preds = self.model(batch)

            pred_parts["contact"].append(preds["contact"].detach().cpu())
            pred_parts["contact_total"].append(preds.get("contact_total", preds["contact"].sum(dim=-1)).detach().cpu())
            pred_parts["preorg"].append(preds["preorg"].detach().cpu())
            pred_parts["delta_total"].append(preds["delta_total"].detach().cpu())

            target_parts["contact"].append(targets["contact"].detach().cpu())
            target_parts["contact_total"].append(targets["contact_total"].detach().cpu())
            target_parts["preorg"].append(targets["preorg"].detach().cpu())
            target_parts["delta_total"].append(targets["delta_total"].detach().cpu())
            target_parts["fg_batch"].append(batch["functional_groups"]["fg_batch"].detach().cpu())
            target_parts["fg_type"].append(batch["functional_groups"]["fg_type"].detach().cpu())

            for key in mask_parts:
                if key in masks:
                    mask_parts[key].append(masks[key].detach().cpu())

        if not pred_parts["contact"]:
            return {}, {}, {}
        all_preds = {key: torch.cat(values, dim=0) for key, values in pred_parts.items()}
        all_targets = {key: torch.cat(values, dim=0) for key, values in target_parts.items()}
        all_masks = {key: torch.cat(values, dim=0) for key, values in mask_parts.items()}
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
        val_contact_r = val_metrics.get("contact_total", {}).get("pearson_r", float("nan")) if val_metrics else float("nan")
        val_delta_mae = val_metrics.get("delta_total", {}).get("mae", float("nan")) if val_metrics else float("nan")
        val_contact_std_ratio = (
            val_metrics.get("diagnostics", {}).get("contact_total", {}).get("pred_target_std_ratio", float("nan"))
            if val_metrics
            else float("nan")
        )
        monitor_path = getattr(self.config, "early_stopping_metric", "contact_total.pearson_r")
        monitor_value = self._get_metric_value(val_metrics, monitor_path) if val_metrics else float("nan")
        print(
            f"epoch {epoch + 1}/{num_epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_contact_r={val_contact_r:.6f} "
            f"val_delta_mae={val_delta_mae:.6f} "
            f"val_contact_std_ratio={val_contact_std_ratio:.6f} "
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

            contact_pred = preds["contact"].detach().cpu()
            contact_total_pred = preds.get("contact_total", preds["contact"].sum(dim=-1)).detach().cpu()
            preorg_pred = preds["preorg"].detach().cpu()
            delta_total_pred = preds["delta_total"].detach().cpu()

            contact_target = targets["contact"].detach().cpu()
            contact_total_target = targets["contact_total"].detach().cpu()
            preorg_target = targets["preorg"].detach().cpu()
            aux_elec_target = targets["aux_elec"].detach().cpu()
            aux_vdw_target = targets["aux_vdw"].detach().cpu()
            delta_total_target = targets["delta_total"].detach().cpu()

            fg_mask = masks["fg"].detach().cpu()
            contact_mask = masks["contact"].detach().cpu()
            contact_total_mask = masks["contact_total"].detach().cpu()
            preorg_mask = masks["preorg"].detach().cpu()
            aux_elec_mask = masks["aux_elec"].detach().cpu()
            aux_vdw_mask = masks["aux_vdw"].detach().cpu()
            delta_total_mask = masks["delta_total"].detach().cpu()

            for complex_idx, pdb_id in enumerate(metadata["pdb_ids"]):
                fg_lo = int(fg_ptr[complex_idx].item())
                fg_hi = int(fg_ptr[complex_idx + 1].item())
                ligand_id = metadata["ligand_ids"][complex_idx]
                smiles = metadata["smiles"][complex_idx]

                complex_rows.append(
                    {
                        "pdb_id": pdb_id,
                        "ligand_id": ligand_id,
                        "smiles": smiles,
                        "n_fg": fg_hi - fg_lo,
                        "pred_delta_total": float(delta_total_pred[complex_idx].item()),
                        "target_delta_total": float(delta_total_target[complex_idx].item()),
                        "mask_delta_total": bool(delta_total_mask[complex_idx].item()),
                        "pred_contact_sum": float(contact_total_pred[fg_lo:fg_hi].sum().item()),
                        "pred_contact_components_sum": float(contact_pred[fg_lo:fg_hi].sum().item()),
                        "pred_preorg_sum": float(preorg_pred[fg_lo:fg_hi].sum().item()),
                    }
                )

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
                            "pred_vdw": float(contact_pred[global_fg_idx, 0].item()),
                            "pred_elec": float(contact_pred[global_fg_idx, 1].item()),
                            "pred_polar": float(contact_pred[global_fg_idx, 2].item()),
                            "pred_nonpolar": float(contact_pred[global_fg_idx, 3].item()),
                            "pred_contact_total": float(contact_total_pred[global_fg_idx].item()),
                            "pred_contact_components_sum": float(contact_pred[global_fg_idx].sum().item()),
                            "pred_preorg": float(preorg_pred[global_fg_idx, 0].item()),
                            "pred_total": float(contact_total_pred[global_fg_idx].item() + preorg_pred[global_fg_idx, 0].item()),
                            "target_vdw": float(contact_target[global_fg_idx, 0].item()),
                            "target_elec": float(contact_target[global_fg_idx, 1].item()),
                            "target_polar": float(contact_target[global_fg_idx, 2].item()),
                            "target_nonpolar": float(contact_target[global_fg_idx, 3].item()),
                            "target_contact_total": float(contact_total_target[global_fg_idx].item()),
                            "target_preorg": float(preorg_target[global_fg_idx, 0].item()),
                            "target_aux_elec": float(aux_elec_target[global_fg_idx].item()),
                            "target_aux_vdw": float(aux_vdw_target[global_fg_idx].item()),
                            "mask_fg": bool(fg_mask[global_fg_idx].item()),
                            "mask_contact": bool(contact_mask[global_fg_idx].item()),
                            "mask_contact_total": bool(contact_total_mask[global_fg_idx].item()),
                            "mask_preorg": bool(preorg_mask[global_fg_idx].item()),
                            "mask_aux_elec": bool(aux_elec_mask[global_fg_idx].item()),
                            "mask_aux_vdw": bool(aux_vdw_mask[global_fg_idx].item()),
                        }
                    )
        return fg_rows, complex_rows
