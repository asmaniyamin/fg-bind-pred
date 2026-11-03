#!/usr/bin/env python3
# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""
Build and optionally execute a model1 training grid.

Covers all 12 ablation axes from the early-plateau investigation plus a few
loss-function and hyperparameter extras:

  1. Loss ablations       --task-mode, --delta-task-weight, --loss-type
                          --huber-beta, --charbonnier-eps, --tukey-c, --quantile-q
  2. LR + scheduler       --lr, --scheduler-type, --warmup-steps, --min-lr
  3. Checkpoint metric    --early-stopping-metric, --early-stopping-mode,
                          --early-stopping-patience
  4. Freezing / curriculum--freeze-module, --freeze-epochs
  5. Architecture size    --hidden-dim, --num-ligand-gnn-layers,
                          --num-fg-attn-layers, --num-fg-global-attn-layers
  6. Branch ablations     --use-aev/--no-use-aev,
                          --use-cross-attention/--no-use-cross-attention
  7. Data filtering       --min-fg-count, --exclude-zero-fg/--no-exclude-zero-fg
  8. Target clipping      --contact-target-clip, --delta-target-clip,
                          --max-abs-delta-total
  9. Batching             --batch-size, --gradient-accumulation-steps
 10. Regularization       --dropout, --weight-decay, --grad-clip
 11. Aux ranking loss     --rank-loss-weight, --rank-loss-margin
 12. Seeds                --seed (repeat for cross-seed variance)

Every axis is a repeatable flag. Passing the flag multiple times sweeps that
axis. The script takes the cartesian product across all axes that were set.

Defaults are tuned for a fast triage run: 30 epochs, max 100 PDB cases per
training run, short patience.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import time
from pathlib import Path

LOSS_CHOICES = ("huber", "mse", "mae", "log_cosh", "charbonnier", "tukey", "quantile", "ccc")
TASK_MODE_CHOICES = ("auto", "full", "contact_only", "contact_delta")
SCHEDULER_CHOICES = ("cosine_warmup", "constant_warmup", "plateau", "none")
ES_MODE_CHOICES = ("max", "min")
BOOL_CHOICES = ("true", "false")
FREEZE_MODULE_CHOICES = (
    "encoders",
    "ligand_gnn",
    "fg_pool",
    "fg_context",
    "cross_attention",
    "contact_head",
    "preorg_head",
    "delta_head",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or run a model1 experiment grid.")
    # ---- environment / I/O ------------------------------------------------
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-prefix", default="m1grid")
    parser.add_argument("--conda-env", default="openmm_env")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plot-style", default="science+ieee")
    # ---- shared training defaults ----------------------------------------
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-cases", type=int, default=100,
                        help="If >0, cap discovered PDB IDs to this many. 0 = unlimited.")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-metric", default="contact_total.pearson_r")
    parser.add_argument("--early-stopping-mode", choices=ES_MODE_CHOICES, default="max")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--rank-loss-margin", type=float, default=0.0)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Optional grouped data split seed passed through to train.py.")
    parser.add_argument("--repeat-splits", type=int, default=1,
                        help="Run repeated grouped holdout splits inside each train.py invocation.")
    parser.add_argument("--split-seed-stride", type=int, default=1,
                        help="Increment applied to split seed between repeated grouped holdout splits.")
    # ---- axes (every one is repeatable) ----------------------------------
    # 1. Loss ablations
    parser.add_argument("--task-mode", action="append", default=None, choices=list(TASK_MODE_CHOICES))
    parser.add_argument("--delta-task-weight", action="append", type=float, default=None)
    parser.add_argument("--loss-type", action="append", default=None, choices=list(LOSS_CHOICES))
    parser.add_argument("--charbonnier-eps", action="append", type=float, default=None)
    parser.add_argument("--tukey-c", action="append", type=float, default=None)
    parser.add_argument("--quantile-q", action="append", type=float, default=None)
    # 2. LR + scheduler
    parser.add_argument("--lr", action="append", type=float, default=None)
    parser.add_argument("--scheduler-type", action="append", default=None, choices=list(SCHEDULER_CHOICES))
    # 4. Freezing / curriculum.  Use "none" for no freezing, or colon-separated module list (e.g. "ligand_gnn:fg_pool").
    parser.add_argument("--freeze", action="append", default=None,
                        help='Either "none" or colon-separated module names. Repeatable.')
    # 5. Architecture size
    parser.add_argument("--hidden-dim", action="append", type=int, default=None)
    parser.add_argument("--num-ligand-gnn-layers", action="append", type=int, default=None)
    parser.add_argument("--num-fg-attn-layers", action="append", type=int, default=None)
    parser.add_argument("--num-fg-global-attn-layers", action="append", type=int, default=None)
    # 6. Branch ablations
    parser.add_argument("--use-aev", action="append", default=None, choices=list(BOOL_CHOICES))
    parser.add_argument("--use-cross-attention", action="append", default=None, choices=list(BOOL_CHOICES))
    # 7. Data filtering
    parser.add_argument("--min-fg-count", action="append", type=int, default=None)
    parser.add_argument("--exclude-zero-fg", action="append", default=None, choices=list(BOOL_CHOICES))
    # 8. Target clipping
    parser.add_argument("--contact-target-clip", action="append", type=float, default=None)
    parser.add_argument("--delta-target-clip", action="append", type=float, default=None)
    parser.add_argument("--max-abs-delta-total", action="append", type=float, default=None)
    # 9. Batching
    parser.add_argument("--batch-size", action="append", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", action="append", type=int, default=None)
    # 10. Regularization
    parser.add_argument("--dropout", action="append", type=float, default=None)
    parser.add_argument("--weight-decay", action="append", type=float, default=None)
    parser.add_argument("--grad-clip", action="append", type=float, default=None)
    # 11. Aux ranking loss
    parser.add_argument("--rank-loss-weight", action="append", type=float, default=None)
    # 12. Seeds
    parser.add_argument("--seed", action="append", type=int, default=None)
    # ---- execution -------------------------------------------------------
    parser.add_argument("--execute", action="store_true",
                        help="Run the commands instead of only printing them.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep running remaining commands if one fails (only with --execute).")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Optional cap on the number of generated runs.")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Number of training runs to launch concurrently. Each run's stdout goes to "
                             "<output-root>/runs/<run-name>/train.log when --max-parallel > 1.")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Seconds between poll cycles when --max-parallel > 1.")
    parser.add_argument("--threads-per-run", type=int, default=0,
                        help="If >0, cap each run's CPU thread pools via OMP/MKL/OPENBLAS/NUMEXPR "
                             "and torch.set_num_threads. Use to keep max-parallel * threads-per-run "
                             "under your machine's thread budget.")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Additional raw arguments appended to every training command.")
    return parser.parse_args(argv)


def _dedupe(seq):
    out = []
    seen = set()
    for item in seq:
        key = item if not isinstance(item, list) else tuple(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _axis(values, default):
    """Return deduped axis values, falling back to a single default if not set."""
    if values is None:
        return [default]
    if not values:
        return [default]
    return _dedupe(values)


def _bool_flag(name: str, value: str) -> list[str]:
    """Turn a "true"/"false" choice into the BooleanOptionalAction CLI form."""
    if value == "true":
        return [f"--{name}"]
    return [f"--no-{name}"]


def _freeze_args(value: str) -> list[str]:
    if value == "none" or not value:
        return []
    modules = [m for m in value.split(":") if m]
    out: list[str] = []
    for module in modules:
        if module not in FREEZE_MODULE_CHOICES:
            raise ValueError(f"Unknown freeze module {module!r}; allowed: {FREEZE_MODULE_CHOICES}")
        out.extend(["--freeze-module", module])
    return out


def _slug_token(value, prefix: str) -> str:
    s = str(value)
    s = s.replace("/", "_").replace(":", "+").replace(" ", "")
    return f"{prefix}{s}"


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    axis_seed = _axis(args.seed, 42)
    axis_lr = _axis(args.lr, 1e-4)
    axis_task_mode = _axis(args.task_mode, "auto")
    axis_scheduler = _axis(args.scheduler_type, "cosine_warmup")
    axis_delta_w = _axis(args.delta_task_weight, 0.25)
    axis_loss = _axis(args.loss_type, "huber")
    axis_charb = _axis(args.charbonnier_eps, 1e-3)
    axis_tukey = _axis(args.tukey_c, 4.685)
    axis_quant = _axis(args.quantile_q, 0.5)
    axis_rank_w = _axis(args.rank_loss_weight, 0.0)
    axis_freeze = _axis(args.freeze, "none")
    axis_hidden = _axis(args.hidden_dim, None)
    axis_n_gnn = _axis(args.num_ligand_gnn_layers, None)
    axis_n_fg_attn = _axis(args.num_fg_attn_layers, None)
    axis_n_fg_global = _axis(args.num_fg_global_attn_layers, None)
    axis_use_aev = _axis(args.use_aev, None)
    axis_use_xattn = _axis(args.use_cross_attention, None)
    axis_min_fg = _axis(args.min_fg_count, None)
    axis_excl_zero = _axis(args.exclude_zero_fg, None)
    axis_clip_contact = _axis(args.contact_target_clip, None)
    axis_clip_delta = _axis(args.delta_target_clip, None)
    axis_max_abs_delta = _axis(args.max_abs_delta_total, None)
    axis_batch = _axis(args.batch_size, 4)
    axis_grad_acc = _axis(args.gradient_accumulation_steps, None)
    axis_dropout = _axis(args.dropout, None)
    axis_wd = _axis(args.weight_decay, None)
    axis_grad_clip = _axis(args.grad_clip, None)

    axes = [
        ("seed", axis_seed),
        ("lr", axis_lr),
        ("task", axis_task_mode),
        ("sched", axis_scheduler),
        ("dw", axis_delta_w),
        ("loss", axis_loss),
        ("cheps", axis_charb),
        ("tkc", axis_tukey),
        ("qq", axis_quant),
        ("rw", axis_rank_w),
        ("frz", axis_freeze),
        ("hd", axis_hidden),
        ("ngnn", axis_n_gnn),
        ("nfg", axis_n_fg_attn),
        ("nfgg", axis_n_fg_global),
        ("aev", axis_use_aev),
        ("xatt", axis_use_xattn),
        ("minfg", axis_min_fg),
        ("ezfg", axis_excl_zero),
        ("cc", axis_clip_contact),
        ("dc", axis_clip_delta),
        ("mad", axis_max_abs_delta),
        ("bs", axis_batch),
        ("ga", axis_grad_acc),
        ("dp", axis_dropout),
        ("wd", axis_wd),
        ("gc", axis_grad_clip),
    ]

    combos = list(itertools.product(*(values for _, values in axes)))
    if args.max_runs > 0:
        combos = combos[: args.max_runs]

    commands: list[list[str]] = []
    extra_split: list[str] = []
    for extra in args.extra_arg:
        extra_split.extend(shlex.split(extra))

    for combo in combos:
        (seed, lr, task_mode, scheduler, dw, loss, cheps, tkc, qq, rw, frz,
         hd, ngnn, nfg, nfgg, aev, xatt, minfg, ezfg, cc, dc, mad, bs, ga, dp, wd, gc) = combo

        slug_parts = [args.run_prefix]
        # only put non-default axes in the run name to keep names short
        slug_parts.append(_slug_token(seed, "s"))
        slug_parts.append(_slug_token(f"{lr:g}", "lr"))
        slug_parts.append(_slug_token(task_mode, ""))
        slug_parts.append(_slug_token(scheduler, ""))
        slug_parts.append(_slug_token(f"{dw:g}", "dw"))
        slug_parts.append(_slug_token(loss, "L"))
        if rw and float(rw) > 0:
            slug_parts.append(_slug_token(f"{rw:g}", "rw"))
        if frz and frz != "none":
            slug_parts.append(_slug_token(frz.replace(":", "+"), "frz"))
        if hd is not None: slug_parts.append(_slug_token(hd, "hd"))
        if ngnn is not None: slug_parts.append(_slug_token(ngnn, "lg"))
        if nfg is not None: slug_parts.append(_slug_token(nfg, "fg"))
        if nfgg is not None: slug_parts.append(_slug_token(nfgg, "fgg"))
        if aev is not None: slug_parts.append(_slug_token(aev, "aev"))
        if xatt is not None: slug_parts.append(_slug_token(xatt, "xa"))
        if minfg is not None: slug_parts.append(_slug_token(minfg, "mfg"))
        if ezfg is not None: slug_parts.append(_slug_token(ezfg, "ez"))
        if cc is not None: slug_parts.append(_slug_token(f"{cc:g}", "cc"))
        if dc is not None: slug_parts.append(_slug_token(f"{dc:g}", "dc"))
        if mad is not None: slug_parts.append(_slug_token(f"{mad:g}", "mad"))
        if bs != 4: slug_parts.append(_slug_token(bs, "bs"))
        if ga is not None: slug_parts.append(_slug_token(ga, "ga"))
        if dp is not None: slug_parts.append(_slug_token(f"{dp:g}", "dp"))
        if wd is not None: slug_parts.append(_slug_token(f"{wd:g}", "wd"))
        if gc is not None: slug_parts.append(_slug_token(f"{gc:g}", "gc"))
        if args.split_seed is not None: slug_parts.append(_slug_token(args.split_seed, "ss"))
        if args.repeat_splits != 1: slug_parts.append(_slug_token(args.repeat_splits, "rs"))
        run_name = "_".join(slug_parts)

        cmd: list[str] = [
            "conda", "run", "--no-capture-output", "-n", args.conda_env,
            "python", "-u", "-m", "models.model1.train",
            "--data-root", args.data_root,
            "--cache-dir", args.cache_dir,
            "--output-root", args.output_root,
            "--run-name", run_name,
            "--batch-size", str(bs),
            "--epochs", str(args.epochs),
            "--max-cases", str(args.max_cases),
            "--warmup-steps", str(args.warmup_steps),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--early-stopping-metric", args.early_stopping_metric,
            "--early-stopping-mode", args.early_stopping_mode,
            "--min-lr", str(args.min_lr),
            "--device", args.device,
            "--plot-style", args.plot_style,
            "--seed", str(seed),
            "--lr", str(lr),
            "--task-mode", task_mode,
            "--scheduler-type", scheduler,
            "--delta-task-weight", str(dw),
            "--loss-type", loss,
            "--huber-beta", str(args.huber_beta),
            "--charbonnier-eps", str(cheps),
            "--tukey-c", str(tkc),
            "--quantile-q", str(qq),
            "--rank-loss-weight", str(rw),
            "--rank-loss-margin", str(args.rank_loss_margin),
        ]
        cmd.extend(_freeze_args(frz))
        if frz != "none" and args.freeze_epochs > 0:
            cmd.extend(["--freeze-epochs", str(args.freeze_epochs)])
        if hd is not None: cmd.extend(["--hidden-dim", str(hd)])
        if ngnn is not None: cmd.extend(["--num-ligand-gnn-layers", str(ngnn)])
        if nfg is not None: cmd.extend(["--num-fg-attn-layers", str(nfg)])
        if nfgg is not None: cmd.extend(["--num-fg-global-attn-layers", str(nfgg)])
        if aev is not None: cmd.extend(_bool_flag("use-aev", aev))
        if xatt is not None: cmd.extend(_bool_flag("use-cross-attention", xatt))
        if minfg is not None: cmd.extend(["--min-fg-count", str(minfg)])
        if ezfg is not None: cmd.extend(_bool_flag("exclude-zero-fg", ezfg))
        if cc is not None: cmd.extend(["--contact-target-clip", str(cc)])
        if dc is not None: cmd.extend(["--delta-target-clip", str(dc)])
        if mad is not None: cmd.extend(["--max-abs-delta-total", str(mad)])
        if ga is not None: cmd.extend(["--gradient-accumulation-steps", str(ga)])
        if dp is not None: cmd.extend(["--dropout", str(dp)])
        if wd is not None: cmd.extend(["--weight-decay", str(wd)])
        if gc is not None: cmd.extend(["--grad-clip", str(gc)])
        if args.split_seed is not None:
            cmd.extend(["--split-seed", str(args.split_seed)])
        if args.repeat_splits != 1:
            cmd.extend(["--repeat-splits", str(args.repeat_splits)])
        if args.split_seed_stride != 1:
            cmd.extend(["--split-seed-stride", str(args.split_seed_stride)])
        cmd.extend(extra_split)
        commands.append(cmd)
    return commands


def _run_name_from_cmd(cmd: list[str]) -> str:
    try:
        i = cmd.index("--run-name")
        return cmd[i + 1]
    except ValueError:
        return "unknown"


def _open_log_file(cmd: list[str], output_root: str) -> tuple[Path, int]:
    run_name = _run_name_from_cmd(cmd)
    run_dir = Path(output_root) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    return log_path, fd


def _thread_env(threads_per_run: int) -> dict[str, str]:
    env = os.environ.copy()
    if threads_per_run > 0:
        n = str(threads_per_run)
        env["OMP_NUM_THREADS"] = n
        env["MKL_NUM_THREADS"] = n
        env["OPENBLAS_NUM_THREADS"] = n
        env["NUMEXPR_NUM_THREADS"] = n
        env["VECLIB_MAXIMUM_THREADS"] = n
        env["TORCH_NUM_THREADS"] = n
    return env


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_commands(args)
    print(f"# {len(commands)} run(s) generated")
    cwd = Path(__file__).resolve().parents[1]
    if not args.execute:
        for cmd in commands:
            print(" ".join(shlex.quote(part) for part in cmd))
        return 0
    if args.max_parallel <= 1:
        return _run_sequential(commands, cwd, args)
    return _run_parallel(commands, cwd, args)


def _run_sequential(commands: list[list[str]], cwd: Path, args: argparse.Namespace) -> int:
    failed: list[int] = []
    env = _thread_env(args.threads_per_run)
    for idx, cmd in enumerate(commands):
        line = " ".join(shlex.quote(part) for part in cmd)
        print(line, flush=True)
        print(f"# [run {idx + 1}/{len(commands)}] starting", flush=True)
        result = subprocess.run(cmd, cwd=cwd, env=env)
        if result.returncode != 0:
            failed.append(idx)
            print(f"# [run {idx + 1}/{len(commands)}] FAILED rc={result.returncode}", flush=True)
            if not args.continue_on_error:
                return result.returncode
        else:
            print(f"# [run {idx + 1}/{len(commands)}] ok", flush=True)
    if failed:
        print(f"# {len(failed)} run(s) failed: {failed}", flush=True)
        return 1
    return 0


def _run_parallel(commands: list[list[str]], cwd: Path, args: argparse.Namespace) -> int:
    pending = list(enumerate(commands))
    active: list[tuple[int, subprocess.Popen, int, Path, float]] = []
    failed: list[int] = []
    total = len(commands)
    completed = 0

    env = _thread_env(args.threads_per_run)

    def launch_one():
        idx, cmd = pending.pop(0)
        log_path, fd = _open_log_file(cmd, args.output_root)
        run_name = _run_name_from_cmd(cmd)
        print(f"# [{idx + 1}/{total}] launch {run_name} -> {log_path}", flush=True)
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=fd, stderr=subprocess.STDOUT, env=env)
        active.append((idx, proc, fd, log_path, time.time()))

    while pending and len(active) < args.max_parallel:
        launch_one()

    while active:
        still_running: list[tuple[int, subprocess.Popen, int, Path, float]] = []
        for idx, proc, fd, log_path, t0 in active:
            rc = proc.poll()
            if rc is None:
                still_running.append((idx, proc, fd, log_path, t0))
                continue
            os.close(fd)
            completed += 1
            elapsed = time.time() - t0
            tag = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"# [{idx + 1}/{total}] {tag} ({elapsed:.0f}s) {log_path}", flush=True)
            if rc != 0:
                failed.append(idx)
                if not args.continue_on_error:
                    for _, p, f, _, _ in still_running:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                        os.close(f)
                    return rc
        active = still_running
        while pending and len(active) < args.max_parallel:
            launch_one()
        if active:
            time.sleep(args.poll_interval)

    print(f"# done. {completed} run(s) completed, {len(failed)} failed.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
