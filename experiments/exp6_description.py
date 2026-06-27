"""
EXP 6 — Text description source ablation.

Hypothesis: Qwen-generated descriptions produce better TRELLIS appearance than
a generic "a {category}" fallback, because they carry style and material cues.

Two-phase design (same layout, different prompts):
  Phase A (desc_qwen)   : normal run, saves layout.json with Qwen description.
  Phase B (desc_generic): reuses Phase A layout.json, overrides description
                          with "a {category}" via --generic_prompt.

Sweep : 3 categories × 5 seeds × 2 conditions = 30 runs.
Fixed : n_examples=5, cfg=7.5, resample_steps=10.

Usage:
  python experiments/exp6_description.py \\
      --partnet_root partnet_mobility \\
      [--output_root outputs/experiments] \\
      [--dry_run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import (
    base_parser,
    build_base_cmd,
    execute_run,
    is_complete,
    print_stats,
    run_dir,
    save_run_config,
)

CATEGORIES = ["Table", "StorageFurniture", "Refrigerator"]
SEEDS = [42, 0, 7, 1, 2]
EXP_DIR = "exp6_description"
FIXED = {"n_examples": 5, "cfg_strength": 7.5, "num_resample_steps": 10}


def _run_phase(
    args,
    output_root: Path,
    condition: str,
    extra_cmd_flags: list[str],
    layout_source_condition: str | None,
) -> dict:
    stats = {"total": 0, "completed": 0, "skipped": 0, "failed": 0}

    for cat in CATEGORIES:
        for seed in SEEDS:
            stats["total"] += 1
            out_dir = run_dir(output_root, EXP_DIR, condition, cat, seed)

            if not args.force and is_complete(out_dir):
                print(f"  skip  {EXP_DIR}/{condition}/{cat}/seed{seed}")
                stats["skipped"] += 1
                continue

            # For Phase B, check that Phase A's layout exists
            layout_json: Path | None = None
            if layout_source_condition is not None:
                layout_json = run_dir(
                    output_root, EXP_DIR, layout_source_condition, cat, seed
                ) / "layout.json"
                if not layout_json.exists():
                    print(f"  SKIP (Phase A layout missing): {layout_json}")
                    stats["skipped"] += 1
                    continue

            print(f"\n→ {EXP_DIR} | {condition} | {cat} | seed={seed}")

            config = {
                "exp_dir": EXP_DIR,
                "condition": condition,
                "category": cat,
                "seed": seed,
                "steps": args.steps,
                **FIXED,
            }
            if layout_json is not None:
                config["layout_json"] = str(layout_json)
            save_run_config(out_dir, config)

            cmd = build_base_cmd(args, cat, seed, out_dir)
            cmd.extend(extra_cmd_flags)
            if layout_json is not None:
                cmd += ["--layout_json", str(layout_json)]

            ok = execute_run(cmd, out_dir, args.dry_run)
            stats["completed" if ok else "failed"] += 1

    return stats


def _add(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def main():
    args = base_parser("EXP 6: Text description source ablation").parse_args()
    output_root = Path(args.output_root)

    print("=== Phase A: Qwen-generated description ===")
    stats_a = _run_phase(
        args, output_root,
        condition="desc_qwen",
        extra_cmd_flags=[],
        layout_source_condition=None,
    )
    print_stats(stats_a)

    print("\n=== Phase B: Generic prompt (reusing Phase A layouts) ===")
    stats_b = _run_phase(
        args, output_root,
        condition="desc_generic",
        extra_cmd_flags=["--generic_prompt"],
        layout_source_condition="desc_qwen",
    )
    print_stats(stats_b)

    print("\n=== EXP 6 total ===")
    print_stats(_add(stats_a, stats_b))


if __name__ == "__main__":
    main()
