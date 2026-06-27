"""
EXP 4 — CFG strength ablation.

Hypothesis: low CFG → coherent geometry but weak text adherence; high CFG →
follows the text description but may conflict with the spatial prior, reducing
Layout IoU.

Sweep : cfg ∈ {3.0, 5.0, 7.5, 10.0} × 2 categories × 3 seeds = 24 runs.
Fixed : n_examples=5, resample_steps=10.

Usage:
  python experiments/exp4_cfg.py \\
      --partnet_root partnet_mobility \\
      [--output_root outputs/experiments] \\
      [--dry_run]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import base_parser, print_stats, run_sweep

CATEGORIES = ["Table", "Refrigerator"]
SEEDS = [42, 0, 7]
CFG_VALUES = [3.0, 5.0, 7.5, 10.0]


def main():
    args = base_parser("EXP 4: CFG strength ablation").parse_args()

    conditions = [(f"cfg{cfg}", {"--cfg_strength": cfg}) for cfg in CFG_VALUES]
    stats = run_sweep(
        args,
        exp_dir="exp4_cfg",
        conditions=conditions,
        categories=CATEGORIES,
        seeds=SEEDS,
        extra_config={"n_examples": 5, "num_resample_steps": 10},
    )
    print_stats(stats)


if __name__ == "__main__":
    main()
