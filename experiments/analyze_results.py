"""
Aggregate all experiment results into a CSV and print per-experiment summary tables.

Walks --output_root for run_config.json + metrics.json pairs.
Also reads layout.json to compute layout quality stats (num_parts,
frac_inbounds, pairwise_overlap_frac) useful for EXP 2 and EXP 5.

Usage:
  python experiments/analyze_results.py \\
      [--output_root outputs/experiments] \\
      [--csv_out     outputs/results.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Layout quality helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_overlap_volume(bmin1, bmax1, bmin2, bmax2) -> float:
    vol = 1.0
    for i in range(3):
        vol *= max(0.0, min(bmax1[i], bmax2[i]) - max(bmin1[i], bmin2[i]))
    return vol


def _layout_stats(layout_path: Path) -> dict:
    with open(layout_path) as f:
        layout = json.load(f)
    parts = layout.get("parts", [])
    n = len(parts)
    if n == 0:
        return {"num_parts": 0, "frac_inbounds": 0.0, "pairwise_overlap_frac": 0.0}

    # Fraction of all bbox coordinates within [-0.5, 0.5]
    all_coords = []
    for p in parts:
        all_coords.extend(p["bbox_min"] + p["bbox_max"])
    frac_inbounds = sum(1 for v in all_coords if -0.5 <= v <= 0.5) / len(all_coords)

    # Fraction of part pairs with any AABB overlap
    pairs = n * (n - 1) // 2
    if pairs == 0:
        return {"num_parts": n, "frac_inbounds": round(frac_inbounds, 4),
                "pairwise_overlap_frac": 0.0}
    overlapping = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if _bbox_overlap_volume(
            parts[i]["bbox_min"], parts[i]["bbox_max"],
            parts[j]["bbox_min"], parts[j]["bbox_max"],
        ) > 1e-6
    )
    return {
        "num_parts": n,
        "frac_inbounds": round(frac_inbounds, 4),
        "pairwise_overlap_frac": round(overlapping / pairs, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Row collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_rows(output_root: Path) -> list[dict]:
    rows = []
    for cfg_path in sorted(output_root.rglob("run_config.json")):
        run_d = cfg_path.parent
        metrics_path = run_d / "metrics.json"
        layout_path = run_d / "layout.json"

        with open(cfg_path) as f:
            row: dict = json.load(f)
        row["run_dir"] = str(run_d)

        metric_keys = [
            "layout_iou", "layout_precision", "layout_recall",
            "n_components", "largest_component_frac", "surface_to_volume",
            "chamfer_mean", "chamfer_min",
        ]
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            for k in metric_keys:
                row[k] = m.get(k)
        else:
            for k in metric_keys:
                row[k] = None

        if layout_path.exists():
            try:
                row.update(_layout_stats(layout_path))
            except Exception:
                row.update({"num_parts": None, "frac_inbounds": None,
                             "pairwise_overlap_frac": None})
        else:
            row.update({"num_parts": None, "frac_inbounds": None,
                         "pairwise_overlap_frac": None})

        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(vals: list) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "  N/A             "
    return f"{np.mean(v):8.4f} ±{np.std(v):6.4f}"


def print_experiment_table(
    rows: list[dict],
    exp_prefix: str,
    metrics: list[str],
    title: str,
) -> None:
    exp_rows = [r for r in rows if str(r.get("exp_dir", "")).startswith(exp_prefix)]
    if not exp_rows:
        return

    conditions = sorted(set(r["condition"] for r in exp_rows))
    col_w = 20

    print(f"\n{'═'*60}")
    print(f"  {title}  ({len(exp_rows)} runs)")
    print(f"{'─'*60}")
    header = f"  {'condition':<{col_w}}" + "".join(f"  {m:<{col_w}}" for m in metrics)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for cond in conditions:
        cond_rows = [r for r in exp_rows if r["condition"] == cond]
        line = f"  {cond:<{col_w}}"
        for m in metrics:
            vals = [r.get(m) for r in cond_rows]
            line += f"  {_fmt(vals):<{col_w}}"
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print("No rows collected — nothing to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    print(f"\nCSV saved → {path}  ({len(rows)} rows, {len(keys)} columns)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Aggregate experiment results")
    p.add_argument("--output_root", default="outputs/experiments")
    p.add_argument("--csv_out",     default="outputs/results.csv")
    args = p.parse_args()

    output_root = Path(args.output_root)
    if not output_root.exists():
        print(f"output_root not found: {output_root}")
        return

    rows = collect_rows(output_root)
    print(f"Found {len(rows)} completed run(s) under {output_root}")

    core = ["layout_iou", "chamfer_mean", "chamfer_min"]
    layout_q = ["num_parts", "frac_inbounds", "pairwise_overlap_frac"]
    coherence = ["layout_precision", "layout_recall",
                 "n_components", "largest_component_frac", "surface_to_volume"]

    print_experiment_table(rows, "exp1", core,
                           "EXP 1 — Category breadth")
    print_experiment_table(rows, "exp2", core + ["num_parts"],
                           "EXP 2 — n_examples ablation")
    print_experiment_table(rows, "exp3", core,
                           "EXP 3 — RePaint resample steps (U)")
    print_experiment_table(rows, "exp4", core,
                           "EXP 4 — CFG strength ablation")
    print_experiment_table(rows, "exp5", core + layout_q,
                           "EXP 5 — Thinking mode ablation")
    print_experiment_table(rows, "exp6", core,
                           "EXP 6 — Text description ablation")
    print_experiment_table(rows, "exp7", core + coherence,
                           "EXP 7 — RePaint masking-schedule ablation")

    save_csv(rows, Path(args.csv_out))


if __name__ == "__main__":
    main()
