"""
EXP 5 — Qwen3 thinking mode ablation.

Hypothesis: thinking mode (enable_thinking=True) produces layouts with fewer
overlapping boxes and better-placed coordinates, translating to higher Layout
IoU and lower Chamfer distance.  Also checks retry rate via run.log.

Sweep : {thinking_on, thinking_off} × 3 categories × 5 seeds = 30 runs.
Fixed : n_examples=5, cfg=7.5, resample_steps=10.

Usage:
  python experiments/exp5_thinking.py \\
      --partnet_root partnet_mobility \\
      [--output_root outputs/experiments] \\
      [--dry_run]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.common import base_parser, print_stats, run_sweep

CATEGORIES = ["Table", "StorageFurniture", "Refrigerator"]
SEEDS = [42, 0, 7, 1, 2]


def main():
    args = base_parser("EXP 5: Thinking mode ablation").parse_args()

    conditions = [
        ("thinking_on",  {}),
        ("thinking_off", {"--no_thinking": True}),
    ]
    stats = run_sweep(
        args,
        exp_dir="exp5_thinking",
        conditions=conditions,
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
