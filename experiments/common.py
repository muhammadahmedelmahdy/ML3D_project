"""
Shared utilities for all experiment runner scripts.

Each experiment script imports from here to get:
  - base_parser()       argparse with all common flags
  - run_sweep()         generic loop: conditions × categories × seeds
  - build_base_cmd()    constructs the run_llm_trellis.py command
  - execute_run()       subprocess wrapper with logging
  - helpers             run_dir, is_complete, save_run_config, print_stats
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Path to the main pipeline script (one level up from this file)
PIPELINE = Path(__file__).parent.parent / "run_llm_trellis.py"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--partnet_root", required=True,
                   help="Root of the PartNet-Mobility dataset")
    p.add_argument("--trellis_model", default="microsoft/TRELLIS-text-large",
                   help="TRELLIS model path or HF hub ID")
    p.add_argument("--ss_encoder_path", default=None,
                   help="Sparse-structure encoder path (auto-inferred if omitted)")
    p.add_argument("--qwen_model", default="Qwen/Qwen3-8B",
                   help="Qwen3 model path or HF hub ID")
    p.add_argument("--output_root", default="outputs/experiments",
                   help="Root directory for all experiment outputs")
    p.add_argument("--steps", type=int, default=250,
                   help="TRELLIS denoising steps")
    p.add_argument("--n_refs", type=int, default=5,
                   help="PartNet-Mobility references for Chamfer distance")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if metrics.json already exists")
    return p


def _resolve_ss_encoder(args) -> str:
    if args.ss_encoder_path:
        return args.ss_encoder_path
    return f"{args.trellis_model}/ckpts/ss_enc_conv3d_16l8_fp16"


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_dir(output_root: Path, exp_dir: str, condition: str,
            category: str, seed: int) -> Path:
    return output_root / exp_dir / condition / category / f"seed{seed}"


def is_complete(out_dir: Path) -> bool:
    return (out_dir / "metrics.json").exists()


def save_run_config(out_dir: Path, config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Command builder
# ─────────────────────────────────────────────────────────────────────────────

def build_base_cmd(args, category: str, seed: int, out_dir: Path) -> list:
    return [
        sys.executable, str(PIPELINE),
        "--category",        category,
        "--partnet_root",    args.partnet_root,
        "--trellis_model",   args.trellis_model,
        "--ss_encoder_path", _resolve_ss_encoder(args),
        "--qwen_model",      args.qwen_model,
        "--output_dir",      str(out_dir),
        "--seed",            str(seed),
        "--steps",           str(args.steps),
        "--n_refs",          str(args.n_refs),
        "--formats",         "mesh", "gaussian",
        "--evaluate",
        "--save_layout",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ─────────────────────────────────────────────────────────────────────────────

def execute_run(cmd: list, out_dir: Path, dry_run: bool = False) -> bool:
    if dry_run:
        print("  [dry]", " ".join(str(c) for c in cmd))
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    if proc.returncode == 0:
        print(f"  OK  ({elapsed:.0f}s)  → {out_dir}")
        return True
    print(f"  FAILED (rc={proc.returncode}, {elapsed:.0f}s) — see {log_path}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Generic sweep loop
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    args,
    exp_dir: str,
    conditions: list[tuple[str, dict]],
    categories: list[str],
    seeds: list[int],
    extra_config: dict | None = None,
) -> dict:
    """
    Generic sweep runner used by exp1–exp5.

    Args:
        conditions: List of (condition_label, extra_cli_args) tuples.
                    extra_cli_args maps "--flag" → value, or "--flag" → True
                    for boolean flags.
        extra_config: Extra key/value pairs written verbatim into run_config.json
                      (e.g. fixed hyperparams not swept in this experiment).

    Returns:
        Stats dict: {total, completed, skipped, failed}.
    """
    output_root = Path(args.output_root)
    stats = {"total": 0, "completed": 0, "skipped": 0, "failed": 0}

    for cond_label, extra_args in conditions:
        for cat in categories:
            for seed in seeds:
                stats["total"] += 1
                out_dir = run_dir(output_root, exp_dir, cond_label, cat, seed)

                if not args.force and is_complete(out_dir):
                    print(f"  skip  {exp_dir}/{cond_label}/{cat}/seed{seed}")
                    stats["skipped"] += 1
                    continue

                print(f"\n→ {exp_dir} | {cond_label} | {cat} | seed={seed}")

                # Build run_config.json
                config: dict = {
                    "exp_dir": exp_dir,
                    "condition": cond_label,
                    "category": cat,
                    "seed": seed,
                    "steps": args.steps,
                    **(extra_config or {}),
                }
                for flag, val in extra_args.items():
                    key = flag.lstrip("-").replace("-", "_")
                    config[key] = val
                save_run_config(out_dir, config)

                # Build and run command
                cmd = build_base_cmd(args, cat, seed, out_dir)
                for flag, val in extra_args.items():
                    if isinstance(val, bool):
                        if val:
                            cmd.append(flag)
                    else:
                        cmd += [flag, str(val)]

                ok = execute_run(cmd, out_dir, args.dry_run)
                stats["completed" if ok else "failed"] += 1

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(stats: dict) -> None:
    print(f"\n{'─'*40}")
    print(f"  Total     : {stats['total']}")
    print(f"  Completed : {stats['completed']}")
    print(f"  Skipped   : {stats['skipped']}")
    print(f"  Failed    : {stats['failed']}")
    print(f"{'─'*40}")
