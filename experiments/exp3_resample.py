"""
EXP 3 — RePaint resampling steps ablation.

Hypothesis: higher U → better harmonization at part boundaries → higher
Layout IoU, with diminishing returns past U=10 and increasing wall-clock cost.

Sweep : U ∈ {1, 5, 10, 20} × Table × 3 seeds = 12 runs.
Fixed : n_examples=5, cfg=7.5.

Usage:
  python experiments/exp3_resample.py \\
      --partnet_root partnet_mobility \\
      [--output_root outputs/experiments] \\
      [--dry_run]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import base_parser, print_stats, run_sweep

CATEGORIES = ["Table"]
SEEDS = [42, 0, 7]
U_VALUES = [1, 5, 10, 20]


def main():
    args = base_parser("EXP 3: RePaint resampling steps (U) ablation").parse_args()

    conditions = [(f"u{u}", {"--num_resample_steps": u}) for u in U_VALUES]
    stats = run_sweep(
        args,
        exp_dir="exp3_resample",
        conditions=conditions,
        categories=CATEGORIES,
        seeds=SEEDS,
        extra_config={"n_examples": 5, "cfg_strength": 7.5},
    )
    print_stats(stats)


if __name__ == "__main__":
    main()
