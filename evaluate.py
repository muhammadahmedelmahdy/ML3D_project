"""
Evaluation metrics for the LLM → TRELLIS pipeline.

Metrics
-------
layout_iou
    Voxelizes the generated mesh and computes IoU against the binary
    bounding-box mask produced by the LLM proposer.  Answers: "does the
    generated object respect the proposed part layout?"

chamfer_distance
    Bidirectional Chamfer Distance between two meshes (sampled as point
    clouds).  Answers: "how close is the generated shape to real PartNet-
    Mobility objects?"

evaluate
    Runs both metrics and returns a summary dict.

Standalone usage
----------------
    python evaluate.py \\
        --mesh         outputs/output.obj \\
        --layout_json  outputs/layout.json \\
        --partnet_root /data/PartNet-Mobility \\
        --category     chair \\
        [--n_refs 5] [--n_samples 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# OBJ I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_obj(path: str):
    """Return (vertices [V,3] float32, faces [F,3] int32) from an OBJ file."""
    verts, faces = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                idx = [int(t.split("/")[0]) - 1 for t in tokens]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    v = np.array(verts, np.float32) if verts else np.zeros((0, 3), np.float32)
    f = np.array(faces, np.int32)   if faces else np.zeros((0, 3), np.int32)
    return v, f


def save_mesh_obj(mesh_result, path: str) -> None:
    """
    Save a TRELLIS MeshExtractResult as a Wavefront OBJ file.

    Args:
        mesh_result: trellis.representations.mesh.MeshExtractResult
        path: Output file path (e.g. "outputs/output.obj").
    """
    verts = mesh_result.vertices.cpu().numpy()   # [V, 3]
    faces = mesh_result.faces.cpu().numpy()      # [F, 3]
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def mesh_result_to_o3d(mesh_result):
    """Convert a TRELLIS MeshExtractResult to an open3d TriangleMesh."""
    import open3d as o3d
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        mesh_result.vertices.cpu().numpy().astype(np.float64)
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        mesh_result.faces.cpu().numpy().astype(np.int32)
    )
    return o3d_mesh


# ──────────────────────────────────────────────────────────────────────────────
# Layout IoU
# ──────────────────────────────────────────────────────────────────────────────

def _mesh_to_clipped_o3d(generated_mesh):
    """Load *generated_mesh* (path or MeshExtractResult) as an open3d mesh
    clipped to the valid [-0.5, 0.5]^3 grid range."""
    import open3d as o3d

    if isinstance(generated_mesh, (str, Path)):
        verts, faces = _load_obj(str(generated_mesh))
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    else:
        o3d_mesh = mesh_result_to_o3d(generated_mesh)

    verts = np.asarray(o3d_mesh.vertices)
    verts = np.clip(verts, -0.5 + 1e-6, 0.5 - 1e-6)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    return o3d_mesh


def voxelize_mesh_dense(generated_mesh, reso: int = 64, fill: bool = True) -> np.ndarray:
    """
    Voxelize *generated_mesh* into a dense boolean occupancy grid.

    Open3D's mesh voxelizer only marks voxels intersected by the mesh
    *surface*, i.e. a hollow shell — but the LLM's proposed_mask_3d fills
    boxes solidly. Comparing a shell against a solid mask makes recall
    (and any "% occupied" metric) meaningless, so by default we flood-fill
    the interior to get a solid occupancy volume matching the mask's
    convention. Pass fill=False to get the raw surface-shell voxelization.

    Args:
        generated_mesh: MeshExtractResult, **or** a path to an OBJ file.
        reso: Voxel grid resolution.
        fill: Flood-fill the mesh interior (requires a reasonably watertight
              mesh; default True).

    Returns:
        Bool array [reso, reso, reso].
    """
    import open3d as o3d
    from scipy import ndimage

    o3d_mesh = _mesh_to_clipped_o3d(generated_mesh)
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh,
        voxel_size=1.0 / reso,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    vol = np.zeros((reso, reso, reso), dtype=bool)
    idx = np.array([v.grid_index for v in vg.get_voxels()], dtype=np.int64)
    if len(idx) > 0:
        idx = np.clip(idx, 0, reso - 1)
        vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if fill and vol.any():
        vol = ndimage.binary_fill_holes(vol)
    return vol


def layout_precision_recall(
    generated_mesh,
    proposed_mask_3d: torch.Tensor,
    reso: int = 64,
) -> dict:
    """
    Split layout IoU into precision/recall to separate two failure modes
    that a plain IoU number conflates:

      recall    - fraction of the proposed box volume actually filled by the
                  generated mesh ("did it respect the layout?").
      precision - fraction of the generated volume that falls inside the
                  proposed boxes ("did it leak outside the layout?").

    This distinction matters for the RePaint masking-schedule ablation:
    relaxing the mask (early_stop / *_decay in llm/box_to_voxel + the
    FlowEulerRepaintSampler) is expected to trade some recall for smoother,
    more coherent geometry (see shape_coherence()).

    Args:
        generated_mesh: MeshExtractResult, **or** a path to an OBJ file.
        proposed_mask_3d: Float tensor [1,1,reso,reso,reso] from
                          llm.box_to_voxel.boxes_to_voxel_mask().
        reso: Voxel resolution (default 64).

    Returns:
        dict with keys "precision", "recall", "iou", all floats in [0, 1].
    """
    gen_vol = voxelize_mesh_dense(generated_mesh, reso=reso)
    prop_vol = proposed_mask_3d.squeeze().cpu().numpy() > 0

    intersection = int(np.logical_and(gen_vol, prop_vol).sum())
    gen_count = int(gen_vol.sum())
    prop_count = int(prop_vol.sum())

    if gen_count == 0 and prop_count == 0:
        return {"precision": 1.0, "recall": 1.0, "iou": 1.0}

    union = gen_count + prop_count - intersection
    return {
        "precision": intersection / gen_count if gen_count > 0 else 1.0,
        "recall": intersection / prop_count if prop_count > 0 else 1.0,
        "iou": intersection / (union + 1e-8),
    }


def layout_iou(
    generated_mesh,
    proposed_mask_3d: torch.Tensor,
    reso: int = 64,
) -> float:
    """
    Measure how well the generated mesh covers the LLM-proposed bounding boxes.

    Voxelizes *generated_mesh* at *reso*³ using Open3D and computes IoU
    against the binary *proposed_mask_3d*.  Both tensors must be in the same
    TRELLIS Z-up coordinate space.

    Args:
        generated_mesh: MeshExtractResult from pipeline.decode_slat(), **or**
                        a path to an OBJ file (str/Path).
        proposed_mask_3d: Float tensor [1,1,reso,reso,reso] from
                          llm.box_to_voxel.boxes_to_voxel_mask().
        reso: Voxel resolution (default 64).

    Returns:
        IoU score in [0, 1].
    """
    return layout_precision_recall(generated_mesh, proposed_mask_3d, reso)["iou"]


def shape_coherence(generated_mesh, reso: int = 64) -> dict:
    """
    Geometric coherence diagnostics from the generated occupancy volume.

    Baseline RePaint pastes the known box layout at every denoising step, so
    the model never gets a chance to fuse geometry smoothly across box
    boundaries; the hypothesis behind the masking-schedule ablation is that
    relaxing the mask should reduce that fragmentation. These metrics make
    that concrete without needing to know the box layout at all:

      n_components           - number of 6-connected occupied components.
      largest_component_frac - fraction of occupied voxels in the largest
                                component (1.0 = a single coherent solid;
                                lower means the boxes never fused into one
                                object).
      surface_to_volume      - surface voxel count / occupied voxel count.
                                Higher indicates a jaggier / more fragmented
                                boundary (e.g. blocky seams left by hard
                                masking); lower is smoother.

    Args:
        generated_mesh: MeshExtractResult, **or** a path to an OBJ file.
        reso: Voxel resolution (default 64).

    Returns:
        dict with keys "n_components" (int), "largest_component_frac" (float),
        "surface_to_volume" (float).
    """
    from scipy import ndimage

    vol = voxelize_mesh_dense(generated_mesh, reso=reso)
    occupied = int(vol.sum())
    if occupied == 0:
        return {"n_components": 0, "largest_component_frac": 0.0,
                "surface_to_volume": 0.0}

    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    labels, n_components = ndimage.label(vol, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]  # drop the background label (0)
    largest_component_frac = float(sizes.max() / occupied)

    # A voxel is "surface" if any of its 6 face-neighbors is empty/out of bounds.
    padded = np.pad(vol, 1, mode="constant", constant_values=False)
    neighbor_sum = np.zeros_like(vol, dtype=np.int8)
    for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        neighbor_sum += padded[1 + dx:1 + dx + reso, 1 + dy:1 + dy + reso, 1 + dz:1 + dz + reso]
    surface = vol & (neighbor_sum < 6)
    surface_to_volume = float(surface.sum() / occupied)

    return {
        "n_components": int(n_components),
        "largest_component_frac": round(largest_component_frac, 4),
        "surface_to_volume": round(surface_to_volume, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Chamfer Distance
# ──────────────────────────────────────────────────────────────────────────────

def sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    n: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """
    Area-weighted uniform sampling of *n* points on a triangle mesh surface.

    Args:
        vertices: [V, 3] float array.
        faces:    [F, 3] int array.
        n:        Number of points to sample.
        seed:     RNG seed for reproducibility.

    Returns:
        [n, 3] float32 array of surface points.
    """
    rng = np.random.default_rng(seed)

    if len(faces) == 0:
        # Degenerate mesh — fall back to random vertices
        idx = rng.choice(len(vertices), size=n, replace=True)
        return vertices[idx].astype(np.float32)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    areas = np.maximum(areas, 0.0)
    total = areas.sum()
    if total < 1e-12:
        idx = rng.choice(len(vertices), size=n, replace=True)
        return vertices[idx].astype(np.float32)

    probs = areas / total
    chosen = rng.choice(len(faces), size=n, p=probs, replace=True)

    # Barycentric sampling
    r1 = rng.random(n)
    r2 = rng.random(n)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    pts = (
        u[:, None] * vertices[faces[chosen, 0]]
        + v[:, None] * vertices[faces[chosen, 1]]
        + w[:, None] * vertices[faces[chosen, 2]]
    )
    return pts.astype(np.float32)


def chamfer_distance(
    verts_pred: np.ndarray,
    verts_ref: np.ndarray,
    faces_pred: Optional[np.ndarray] = None,
    faces_ref: Optional[np.ndarray] = None,
    n_samples: int = 10_000,
    seed: int = 0,
) -> float:
    """
    Bidirectional Chamfer Distance between two meshes (or point clouds).

    If *faces_* are provided, points are sampled from the mesh surface via
    area-weighted sampling.  Otherwise the vertex arrays are used directly
    as point clouds.

    Args:
        verts_pred: [V1, 3] generated mesh vertices.
        verts_ref:  [V2, 3] reference mesh vertices.
        faces_pred: [F1, 3] generated mesh faces (optional).
        faces_ref:  [F2, 3] reference mesh faces (optional).
        n_samples:  Points per mesh.
        seed:       RNG seed.

    Returns:
        (mean-squared Chamfer pred→ref + ref→pred) / 2.
    """
    from scipy.spatial import cKDTree

    pts_pred = (
        sample_surface_points(verts_pred, faces_pred, n=n_samples, seed=seed)
        if faces_pred is not None and len(faces_pred) > 0
        else verts_pred
    )
    pts_ref = (
        sample_surface_points(verts_ref, faces_ref, n=n_samples, seed=seed)
        if faces_ref is not None and len(faces_ref) > 0
        else verts_ref
    )

    d_pred_to_ref, _ = cKDTree(pts_ref).query(pts_pred, workers=-1)
    d_ref_to_pred, _ = cKDTree(pts_pred).query(pts_ref, workers=-1)
    return float((np.mean(d_pred_to_ref ** 2) + np.mean(d_ref_to_pred ** 2)) / 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Reference mesh loader (PartNet-Mobility)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """Scale vertices so the bounding box fits in [-0.5, 0.5]³."""
    lo = vertices.min(0)
    hi = vertices.max(0)
    center = (lo + hi) / 2.0
    scale = (hi - lo).max()
    if scale < 1e-8:
        scale = 1.0
    return ((vertices - center) / scale).astype(np.float32)


def _load_partnet_object(obj_dir: Path):
    """
    Load and merge all part meshes from a PartNet-Mobility object directory.

    Supports two common layout variants:
      - Processed:  {obj_dir}/mesh.obj
      - Raw v2:     {obj_dir}/textured_objs/*.obj
      - Raw v1:     {obj_dir}/objs/*.obj

    Returns:
        (vertices [V,3], faces [F,3]) or (None, None) if no geometry found.
    """
    # 1. Pre-processed single mesh
    processed = obj_dir / "mesh.obj"
    if processed.exists():
        v, f = _load_obj(str(processed))
        return (v, f) if len(v) > 0 else (None, None)

    # 2. Raw multi-part layout
    for subdir in ("textured_objs", "objs"):
        part_dir = obj_dir / subdir
        if not part_dir.exists():
            continue
        obj_files = sorted(part_dir.glob("*.obj"))
        if not obj_files:
            continue

        all_v, all_f = [], []
        v_offset = 0
        for of in obj_files:
            v, f = _load_obj(str(of))
            if len(v) == 0:
                continue
            all_v.append(v)
            if len(f) > 0:
                all_f.append(f + v_offset)
            v_offset += len(v)

        if all_v:
            merged_v = np.concatenate(all_v, 0)
            merged_f = np.concatenate(all_f, 0) if all_f else np.zeros((0, 3), np.int32)
            return merged_v, merged_f

    return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    generated_mesh,
    proposed_mask_3d: torch.Tensor,
    partnet_root: str,
    category: str,
    n_refs: int = 5,
    n_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    """
    Evaluate a single generated asset against the LLM layout and PartNet GT.

    Args:
        generated_mesh: MeshExtractResult **or** path to an OBJ file.
        proposed_mask_3d: Float tensor [1,1,64,64,64] (from layout_to_trellis_inputs).
        partnet_root: Root directory of PartNet-Mobility.
        category: Object category (e.g. "chair").
        n_refs: Number of PartNet-Mobility reference objects for Chamfer.
        n_samples: Surface points per mesh for Chamfer distance.
        seed: RNG seed.

    Returns:
        dict with keys:
          layout_iou              – float in [0, 1]
          layout_precision        – float in [0, 1], fraction of generated
                                     volume inside the proposed boxes
          layout_recall           – float in [0, 1], fraction of proposed
                                     box volume filled by the generated mesh
          n_components            – int, connected components in the
                                     generated occupancy volume
          largest_component_frac  – float in [0, 1], fraction of occupied
                                     voxels in the largest component
          surface_to_volume       – float, surface voxel count / occupied
                                     voxel count (jaggedness proxy)
          chamfer_mean  – mean Chamfer distance to n_refs GT objects (or None)
          chamfer_min   – best (lowest) Chamfer distance (or None)
          chamfer_all   – list of per-object Chamfer distances
    """
    results: dict = {}

    # ── 1. Layout IoU / precision / recall ────────────────────────────────────
    print("  Computing layout IoU / precision / recall …")
    layout_metrics = layout_precision_recall(generated_mesh, proposed_mask_3d)
    results["layout_iou"] = layout_metrics["iou"]
    results["layout_precision"] = layout_metrics["precision"]
    results["layout_recall"] = layout_metrics["recall"]
    print(f"    Layout IoU: {layout_metrics['iou']:.4f}  "
          f"(precision={layout_metrics['precision']:.4f}, "
          f"recall={layout_metrics['recall']:.4f})  "
          f"({'good' if layout_metrics['iou'] > 0.4 else 'low — mesh may not follow the proposed layout'})")

    # ── 1b. Shape coherence (fragmentation / seam-jaggedness proxy) ──────────
    print("  Computing shape coherence …")
    coherence = shape_coherence(generated_mesh)
    results["n_components"] = coherence["n_components"]
    results["largest_component_frac"] = coherence["largest_component_frac"]
    results["surface_to_volume"] = coherence["surface_to_volume"]
    print(f"    Components: {coherence['n_components']}  "
          f"largest_frac={coherence['largest_component_frac']:.4f}  "
          f"surface/volume={coherence['surface_to_volume']:.4f}")

    # ── 2. Chamfer vs PartNet-Mobility ground truth ───────────────────────────
    print(f"  Computing Chamfer distance vs {n_refs} PartNet-Mobility references …")

    # Prepare generated mesh vertices/faces
    if isinstance(generated_mesh, (str, Path)):
        gen_v, gen_f = _load_obj(str(generated_mesh))
    else:
        gen_v = generated_mesh.vertices.cpu().numpy()
        gen_f = generated_mesh.faces.cpu().numpy()

    if len(gen_v) == 0:
        print("    Warning: generated mesh has no vertices — skipping Chamfer.")
        results.update(chamfer_mean=None, chamfer_min=None, chamfer_all=[])
        return results

    gen_v_norm = _normalize_vertices(gen_v)

    cat_dir = Path(partnet_root) / category
    if not cat_dir.exists():
        print(f"    Warning: category directory not found at {cat_dir}")
        results.update(chamfer_mean=None, chamfer_min=None, chamfer_all=[])
        return results

    rng = np.random.default_rng(seed)
    obj_dirs = [d for d in cat_dir.iterdir() if d.is_dir()]
    rng.shuffle(obj_dirs)

    chamfers = []
    for obj_dir in obj_dirs:
        if len(chamfers) >= n_refs:
            break
        ref_v, ref_f = _load_partnet_object(obj_dir)
        if ref_v is None or len(ref_v) == 0:
            continue
        ref_v_norm = _normalize_vertices(ref_v)
        cd = chamfer_distance(
            gen_v_norm, ref_v_norm, gen_f, ref_f,
            n_samples=n_samples, seed=seed,
        )
        chamfers.append(cd)

    if chamfers:
        results["chamfer_mean"] = float(np.mean(chamfers))
        results["chamfer_min"] = float(np.min(chamfers))
        results["chamfer_all"] = chamfers
        print(f"    Chamfer mean: {results['chamfer_mean']:.6f}")
        print(f"    Chamfer best: {results['chamfer_min']:.6f}  "
              f"(over {len(chamfers)} references)")
    else:
        print(f"    Warning: no usable reference objects found in {cat_dir}")
        results.update(chamfer_mean=None, chamfer_min=None, chamfer_all=[])

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a generated mesh.")
    p.add_argument("--mesh", required=True,
                   help="Path to the generated mesh OBJ file.")
    p.add_argument("--layout_json", required=True,
                   help="Layout JSON produced by the Qwen proposer.")
    p.add_argument("--partnet_root", required=True,
                   help="Root of the PartNet-Mobility dataset.")
    p.add_argument("--category", required=True,
                   help="Object category (e.g. 'chair').")
    p.add_argument("--n_refs", type=int, default=5,
                   help="Number of PartNet-Mobility references for Chamfer.")
    p.add_argument("--n_samples", type=int, default=10_000,
                   help="Surface points per mesh for Chamfer distance.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reso", type=int, default=64,
                   help="Voxel resolution for layout IoU.")
    p.add_argument("--out_json", default=None,
                   help="Optional path to save the metrics as JSON.")
    return p.parse_args()


def main():
    args = _parse_args()

    # Reconstruct proposed_mask_3d from the layout JSON
    with open(args.layout_json) as f:
        layout = json.load(f)

    from llm.box_to_voxel import boxes_to_voxel_mask
    proposed_mask_3d = boxes_to_voxel_mask(layout["parts"], reso=args.reso)

    print(f"Evaluating: {args.mesh}")
    results = evaluate(
        generated_mesh=args.mesh,
        proposed_mask_3d=proposed_mask_3d,
        partnet_root=args.partnet_root,
        category=args.category,
        n_refs=args.n_refs,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    print("\nSummary:")
    print(f"  layout_iou    = {results['layout_iou']:.4f}")
    print(f"  layout_precision = {results['layout_precision']:.4f}")
    print(f"  layout_recall    = {results['layout_recall']:.4f}")
    print(f"  n_components     = {results['n_components']}")
    print(f"  largest_component_frac = {results['largest_component_frac']:.4f}")
    print(f"  surface_to_volume       = {results['surface_to_volume']:.4f}")
    if results["chamfer_mean"] is not None:
        print(f"  chamfer_mean  = {results['chamfer_mean']:.6f}")
        print(f"  chamfer_min   = {results['chamfer_min']:.6f}")
    else:
        print("  chamfer_mean  = N/A")
        print("  chamfer_min   = N/A")

    if args.out_json:
        out = {k: v for k, v in results.items() if k != "chamfer_all"}
        out["chamfer_all"] = results["chamfer_all"]
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nMetrics saved to {args.out_json}")


if __name__ == "__main__":
    main()
