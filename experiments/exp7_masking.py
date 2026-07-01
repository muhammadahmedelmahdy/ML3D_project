"""
EXP 7 — RePaint masking-schedule ablation.

Baseline RePaint pastes the known box layout into the sample at *every*
denoising step, from t=1 (pure noise) down to t=0 (clean) — the model never
gets a chance to freely reconcile geometry across box boundaries. Hypothesis:
relaxing that constraint over the tail of the trajectory (t < t_stop_mask)
should reduce hard-seam fragmentation (fewer disconnected components, lower
surface_to_volume) at some cost to layout recall, since the model has more
freedom to depart from the proposed boxes.

Conditions:
  full                     - baseline: mask enforced at every step (no relaxation).
  early_stop@{0.1,0.2,0.4}   - mask at full strength above t_stop_mask, then a
                               hard cutoff to 0 below it.
  linear_decay@{0.1,0.2,0.4} - mask strength ramps linearly to 0 below t_stop_mask.
  cosine_decay@{0.1,0.2,0.4} - same, eased with a cosine taper.

Sweep : 10 conditions × Table × 3 seeds = 30 runs.
Fixed : n_examples=5, cfg=7.5, resample_steps=10.

Usage:
  python experiments/exp7_masking.py \\
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
T_STOP_VALUES = [0.1, 0.2, 0.4]
SCHEDULES = ["early_stop", "linear_decay", "cosine_decay"]


def main():
    args = base_parser("EXP 7: RePaint masking-schedule ablation").parse_args()

    conditions = [("full", {})]
    for sched in SCHEDULES:
        for t_stop in T_STOP_VALUES:
            conditions.append((
                f"{sched}@{t_stop}",
                {"--mask_schedule": sched, "--t_stop_mask": t_stop},
            ))

    stats = run_sweep(
        args,
        exp_dir="exp7_masking",
        conditions=conditions,
        categories=CATEGORIES,
        seeds=SEEDS,
        extra_config={"n_examples": 5, "cfg_strength": 7.5, "num_resample_steps": 10},
    )
    print_stats(stats)


if __name__ == "__main__":
    main()
