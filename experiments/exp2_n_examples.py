"""
EXP 2 — Context set size ablation.

Hypothesis: more in-context examples → better Qwen layout proposals (fewer
overlaps, valid coordinates) → higher Layout IoU downstream.

Sweep : n_examples ∈ {1, 3, 5, 10} × 2 categories × 3 seeds = 24 runs.
Fixed : cfg=7.5, resample_steps=10.

Usage:
  python experiments/exp2_n_examples.py \\
      --partnet_root partnet_mobility \\
      [--output_root outputs/experiments] \\
      [--dry_run]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import base_parser, print_stats, run_sweep

CATEGORIES = ["Table", "StorageFurniture"]
SEEDS = [42, 0, 7]
N_VALUES = [1, 3, 5, 10]


def main():
    args = base_parser("EXP 2: n_examples ablation").parse_args()

    conditions = [(f"n{n}", {"--n_examples": n}) for n in N_VALUES]
    stats = run_sweep(
        args,
        exp_dir="exp2_n_examples",
        conditions=conditions,
        categories=CATEGORIES,
        seeds=SEEDS,
        extra_config={"cfg_strength": 7.5, "num_resample_steps": 10},
    )
    print_stats(stats)


if __name__ == "__main__":
    main()
