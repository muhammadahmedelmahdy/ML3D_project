"""
Run Qwen3-8B layout proposal and visualize the proposed bounding boxes.

Steps:
  1. Load PartNet-Mobility context examples for the category.
  2. Prompt Qwen3-8B → JSON layout (parts + bounding boxes).
  3. Render the bounding boxes as a color-coded 3D figure → PNG.
  4. Save layout.json and box skeleton mesh (OBJ).

Usage:
  python propose_and_visualize.py \\
      --category Table \\
      --partnet_root partnet_mobility \\
      --output_dir outputs/layout_preview \\
      [--n_examples 5] [--seed 42] \\
      [--layout_json existing_layout.json]   # skip Qwen, just visualize
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# Box drawing helpers
# ---------------------------------------------------------------------------

def _box_faces(bmin, bmax):
    """Return the 6 faces of an AABB as lists of (x,y,z) quads."""
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    corners = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[0], corners[3], corners[7], corners[4]],  # left
        [corners[1], corners[2], corners[6], corners[5]],  # right
    ]
    return faces


def visualize_layout(layout: dict, save_path: str) -> None:
    """
    Render labeled bounding boxes in TRELLIS Z-up space and save as PNG.

    The LLM outputs boxes in Y-up space; we apply the same axis swap used by
    box_to_voxel.py ([x, y, z]_llm → [x, -z, y]_trellis) so the picture
    matches what TRELLIS will receive.

    Args:
        layout: Validated layout dict with 'description' and 'parts'.
        save_path: Output PNG path.
    """
    from llm.box_to_voxel import _bbox_to_trellis

    parts = layout["parts"]
    n = len(parts)

    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    colors = [cmap(i) for i in range(n)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    legend_patches = []
    for i, part in enumerate(parts):
        tm, tM = _bbox_to_trellis(part["bbox_min"], part["bbox_max"])
        faces = _box_faces(tm, tM)
        rgba = list(colors[i])
        face_color = rgba[:3] + [0.20]   # translucent fill
        edge_color = rgba[:3] + [0.90]   # solid edges

        poly = Poly3DCollection(faces, facecolor=face_color,
                                edgecolor=edge_color, linewidth=0.8)
        ax.add_collection3d(poly)
        legend_patches.append(mpatches.Patch(color=colors[i], label=part["label"]))

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y (TRELLIS)")
    ax.set_zlabel("Z (up)")
    ax.set_title(f"{layout.get('description', '')}", wrap=True, fontsize=9)

    ax.legend(handles=legend_patches, loc="upper left",
              bbox_to_anchor=(1.02, 1), fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Visualization saved → {save_path}")


def save_box_mesh(layout: dict, save_path: str) -> None:
    """Save the box skeleton as a Wavefront OBJ (viewable in any 3D viewer)."""
    from llm.box_to_voxel import boxes_to_mesh
    import open3d as o3d
    mesh = boxes_to_mesh(layout["parts"])
    o3d.io.write_triangle_mesh(save_path, mesh)
    print(f"      Box mesh saved     → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True,
                   help="PartNet-Mobility category, e.g. 'Table'")
    p.add_argument("--partnet_root", default="partnet_mobility")
    p.add_argument("--output_dir", default="outputs/layout_preview")
    p.add_argument("--n_examples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qwen_model", default="Qwen/Qwen3-8B")
    p.add_argument("--layout_json", default=None,
                   help="Skip Qwen and load an existing layout JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build context ─────────────────────────────────────────────────
    if args.layout_json is None:
        print(f"[1/3] Loading PartNet-Mobility context for '{args.category}' …")
        from llm.context_builder import PartNetMobilityContextBuilder
        builder = PartNetMobilityContextBuilder(args.partnet_root)
        examples_str = builder.format_examples_for_prompt(
            args.category, n=args.n_examples, seed=args.seed
        )
        print(f"      {args.n_examples} examples loaded.")
        print(f"\n--- In-context examples sent to Qwen ---\n{examples_str}\n---\n")

    # ── Step 2: Qwen layout proposal ─────────────────────────────────────────
    if args.layout_json is not None:
        print(f"[2/3] Loading layout from {args.layout_json} …")
        with open(args.layout_json) as f:
            layout = json.load(f)
    else:
        print(f"[2/3] Prompting Qwen3-8B …")
        from llm.qwen_proposer import QwenLayoutProposer
        proposer = QwenLayoutProposer(model_name=args.qwen_model)
        layout = proposer.propose(
            category=args.category,
            examples_str=examples_str,
            n_examples=args.n_examples,
        )

    print(f"\nProposed layout:")
    print(f"  Description: {layout['description']}")
    print(f"  Parts ({len(layout['parts'])}):")
    for p in layout["parts"]:
        print(f"    {p['label']:20s}  min={p['bbox_min']}  max={p['bbox_max']}")

    layout_path = str(out_dir / "layout.json")
    with open(layout_path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"\n      Layout saved        → {layout_path}")

    # ── Step 3: Visualize ────────────────────────────────────────────────────
    print(f"[3/3] Rendering bounding boxes …")
    visualize_layout(layout, str(out_dir / "layout_boxes.png"))
    save_box_mesh(layout, str(out_dir / "layout_boxes.obj"))

    print("\nDone.")


if __name__ == "__main__":
    main()
