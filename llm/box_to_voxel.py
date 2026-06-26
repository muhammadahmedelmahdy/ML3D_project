"""
Convert LLM-proposed part bounding boxes into TRELLIS voxel representations.

The LLM outputs bounding boxes in Y-up OBJ space (PartNet-Mobility convention).
TRELLIS uses Z-up (Blender convention).  All functions here apply the axis
transform to_trellis([x, y, z]) = [x, -z, y] before building the mask or mesh.

Produces:
  - structure_mask_3d  : float32 [1,1,64,64,64] — 1 where any box occupies
  - coords             : int32 [M,3]             — (x,y,z) voxel coords
  - mesh               : open3d TriangleMesh      — box skeleton in TRELLIS space
"""

from __future__ import annotations
from typing import List

import numpy as np
import torch


VOXEL_RESO = 64


def _to_trellis(pts: np.ndarray) -> np.ndarray:
    """Convert points from LLM Y-up space to TRELLIS Z-up space.

    Mapping: [x, y, z]_llm → [x, -z, y]_trellis
    """
    return np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=1)


def _world_to_voxel(coord: float, reso: int = VOXEL_RESO) -> int:
    """Map world coordinate in [-0.5, 0.5] to voxel index [0, reso-1]."""
    return int(np.clip((coord + 0.5) * reso, 0, reso - 1))


def _bbox_to_trellis(bmin: list, bmax: list):
    """Return (trellis_min, trellis_max) for a single LLM bbox.

    Negating the Z axis swaps min/max for the TRELLIS Y component:
      TRELLIS_Y_min = -bmax[2],  TRELLIS_Y_max = -bmin[2]
    """
    tm = [bmin[0], -bmax[2], bmin[1]]
    tM = [bmax[0], -bmin[2], bmax[1]]
    # Guard against LLM flipping min/max in any axis
    result_min = [min(a, b) for a, b in zip(tm, tM)]
    result_max = [max(a, b) for a, b in zip(tm, tM)]
    return result_min, result_max


def boxes_to_voxel_mask(
    parts: List[dict],
    reso: int = VOXEL_RESO,
) -> torch.Tensor:
    """
    Rasterize part bounding boxes into a binary voxel mask in TRELLIS space.

    Args:
        parts: List of part dicts with "bbox_min"/"bbox_max" in LLM Y-up space.
        reso: Voxel grid resolution (64 matches TRELLIS).

    Returns:
        Float32 tensor [1, 1, reso, reso, reso] — 1 inside any box, 0 outside.
    """
    mask = np.zeros((reso, reso, reso), dtype=np.float32)

    for part in parts:
        tm, tM = _bbox_to_trellis(part["bbox_min"], part["bbox_max"])

        x0, y0, z0 = [_world_to_voxel(v, reso) for v in tm]
        x1, y1, z1 = [_world_to_voxel(v, reso) for v in tM]

        mask[x0:x1+1, y0:y1+1, z0:z1+1] = 1.0

    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1,1,H,W,D]


def boxes_to_coords(
    parts: List[dict],
    reso: int = VOXEL_RESO,
) -> torch.Tensor:
    """Return occupied voxel coords [M, 3] in TRELLIS space."""
    mask = boxes_to_voxel_mask(parts, reso)
    return torch.argwhere(mask[0, 0] > 0).int()


def boxes_to_mesh(parts: List[dict]):
    """
    Build an open3d TriangleMesh from the bounding boxes in TRELLIS space.

    Args:
        parts: Part list with "bbox_min"/"bbox_max" in LLM Y-up space.

    Returns:
        open3d.geometry.TriangleMesh in TRELLIS Z-up space.
    """
    import open3d as o3d

    combined = o3d.geometry.TriangleMesh()

    for part in parts:
        tm, tM = _bbox_to_trellis(part["bbox_min"], part["bbox_max"])
        tm = np.array(tm, dtype=np.float64)
        tM = np.array(tM, dtype=np.float64)

        extent = np.maximum(tM - tm, 1e-4)
        box = o3d.geometry.TriangleMesh.create_box(
            width=float(extent[0]),
            height=float(extent[1]),
            depth=float(extent[2]),
        )
        box.translate(tm)
        combined += box

    combined.merge_close_vertices(1e-5)
    combined.remove_duplicated_vertices()
    combined.remove_duplicated_triangles()
    return combined


def layout_to_trellis_inputs(
    layout: dict,
    reso: int = VOXEL_RESO,
    device: str = "cuda",
):
    """
    Convert a validated Qwen layout dict into everything run_inpaint() needs.

    Bboxes from the LLM are in Y-up space; the returned mesh and mask are
    both converted to TRELLIS Z-up space via _bbox_to_trellis().

    Returns:
        (prompt, mesh, structure_mask_3d)
    """
    parts = layout["parts"]
    prompt = layout["description"]

    mesh = boxes_to_mesh(parts)
    structure_mask_3d = boxes_to_voxel_mask(parts, reso=reso).to(device)

    return prompt, mesh, structure_mask_3d
