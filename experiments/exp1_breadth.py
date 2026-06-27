"""
EXP 1 — Category breadth.

Hypothesis: the pipeline succeeds across all 7 categories; Layout IoU and
Chamfer distance vary with structural complexity (e.g. Refrigerator > Table).

Sweep : 7 categories × 3 seeds = 21 runs.
Fixed : n_examples=5, cfg=7.5, resample_steps=10.

Usage:
  python experiments/exp1_breadth.py \\
      --partnet_root partnet_mobility \\
      [--output_root outputs/experiments] \\
      [--dry_run]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import base_parser, print_stats, run_sweep

CATEGORIES = [
    "Table", "StorageFurniture", "Refrigerator",
    "Dishwasher", "Oven", "Microwave", "WashingMachine",
]
SEEDS = [42, 0, 7]


def main():
    args = base_parser("EXP 1: Category breadth sweep").parse_args()

    stats = run_sweep(
        args,
        exp_dir="exp1_breadth",
        conditions=[("default", {})],
        categories=CATEGORIES,
        seeds=SEEDS,
        extra_config={
            "n_examples": 5,
            "cfg_strength": 7.5,
            "num_resample_steps": 10,
        },
    )
    print_stats(stats)


if __name__ == "__main__":
    main()
