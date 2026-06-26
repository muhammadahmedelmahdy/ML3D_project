"""
Builds in-context examples from the PartNet-Mobility dataset.

Each example is a compact dict:
  {
    "category": str,
    "parts": [
      {"label": str, "bbox_min": [x,y,z], "bbox_max": [x,y,z]},
      ...
    ]
  }

All bounding boxes are in the TRELLIS normalized coordinate space [-0.5, 0.5].

PartNet-Mobility dataset structure expected at dataset_root:
  {object_id}/
    mobility_v2.json          # part tree
    textured_objs/
      {part_name}_{id}.obj    # one mesh per part link
"""

from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_obj_vertices(obj_path: str) -> Optional[np.ndarray]:
    """Return (N,3) array of vertices from an OBJ file, or None if empty."""
    verts = []
    try:
        with open(obj_path) as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception:
        return None
    return np.array(verts, dtype=np.float32) if verts else None


def _compute_part_bbox(obj_dir: str, link: dict) -> Optional[dict]:
    """
    Compute the axis-aligned bounding box of a single part link.

    Loads all OBJ files whose name starts with the link's name, merges their
    vertices, and returns {bbox_min, bbox_max} or None if no geometry found.
    """
    parts_refs = link.get("parts", [])
    all_verts = []
    for part_ref in parts_refs:
        # typical naming: "{link_name}_{part_id}.obj" or just "{id}.obj"
        candidates = [
            os.path.join(obj_dir, "textured_objs", f"{part_ref['id']}.obj"),
            os.path.join(obj_dir, "textured_objs", f"{link['name']}_{part_ref['id']}.obj"),
        ]
        for path in candidates:
            if os.path.exists(path):
                v = _load_obj_vertices(path)
                if v is not None and len(v):
                    all_verts.append(v)
                break

    if not all_verts:
        return None

    verts = np.concatenate(all_verts, axis=0)
    return {"bbox_min": verts.min(0).tolist(), "bbox_max": verts.max(0).tolist()}


def _normalize_bboxes(parts: List[dict]) -> List[dict]:
    """
    Normalize all part bboxes so the union AABB maps to [-0.5, 0.5]^3,
    matching TRELLIS's coordinate convention.
    """
    if not parts:
        return parts

    all_mins = np.array([p["bbox_min"] for p in parts])
    all_maxs = np.array([p["bbox_max"] for p in parts])
    global_min = all_mins.min(0)
    global_max = all_maxs.max(0)
    center = (global_min + global_max) / 2.0
    scale = (global_max - global_min).max()
    if scale < 1e-8:
        scale = 1.0

    normalized = []
    for p in parts:
        bmin = ((np.array(p["bbox_min"]) - center) / scale).tolist()
        bmax = ((np.array(p["bbox_max"]) - center) / scale).tolist()
        normalized.append({
            "label": p["label"],
            "bbox_min": [round(v, 4) for v in bmin],
            "bbox_max": [round(v, 4) for v in bmax],
        })
    return normalized


# ---------------------------------------------------------------------------
# Per-object loader
# ---------------------------------------------------------------------------

def _load_from_object_json(obj_dir: str, category: str) -> Optional[dict]:
    """
    Load bounding boxes from object.json (PartNet-Mobility v2 format).

    Each node in diffuse_tree has a pre-computed aabb with center + size,
    so no OBJ parsing is needed.
    """
    object_path = os.path.join(obj_dir, "object.json")
    with open(object_path) as f:
        data = json.load(f)

    parts = []
    for node in data.get("diffuse_tree", []):
        aabb = node.get("aabb")
        if not aabb:
            continue
        cx, cy, cz = aabb["center"]
        sx, sy, sz = aabb["size"]
        parts.append({
            "label": node.get("name", "unknown"),
            "bbox_min": [cx - sx / 2, cy - sy / 2, cz - sz / 2],
            "bbox_max": [cx + sx / 2, cy + sy / 2, cz + sz / 2],
        })
    return parts or None


def _load_from_mobility_v2(obj_dir: str) -> Optional[list]:
    """Load bounding boxes from mobility_v2.json by parsing OBJ vertices."""
    mobility_path = os.path.join(obj_dir, "mobility_v2.json")
    with open(mobility_path) as f:
        mobility = json.load(f)

    parts = []
    for link in mobility:
        bbox = _compute_part_bbox(obj_dir, link)
        if bbox is None:
            continue
        parts.append({
            "label": link.get("name", "unknown"),
            "bbox_min": bbox["bbox_min"],
            "bbox_max": bbox["bbox_max"],
        })
    return parts or None


def load_object(obj_dir: str, category: str) -> Optional[dict]:
    """
    Load one PartNet-Mobility object directory and return a context example.

    Tries object.json first (pre-computed AABBs, no OBJ parsing needed),
    then falls back to mobility_v2.json + OBJ vertex loading.

    Returns None if the object cannot be parsed.
    """
    parts = None

    if os.path.exists(os.path.join(obj_dir, "object.json")):
        try:
            parts = _load_from_object_json(obj_dir, category)
        except Exception:
            pass

    if parts is None and os.path.exists(os.path.join(obj_dir, "mobility_v2.json")):
        try:
            parts = _load_from_mobility_v2(obj_dir)
        except Exception:
            pass

    if not parts:
        return None

    parts = _normalize_bboxes(parts)
    return {"category": category, "parts": parts}


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class PartNetMobilityContextBuilder:
    """
    Indexes a PartNet-Mobility dataset root and builds compact in-context
    examples for prompting Qwen.

    Dataset root layout:
        dataset_root/
            {category}/
                {object_id}/
                    mobility_v2.json
                    textured_objs/
                        ...

    Args:
        dataset_root: Path to the PartNet-Mobility dataset.
        max_examples_per_category: Cap on how many objects to load per category.
    """

    def __init__(self, dataset_root: str, max_examples_per_category: int = 50):
        self.dataset_root = Path(dataset_root)
        self.max_per_cat = max_examples_per_category
        # category → list of context example dicts
        self._cache: Dict[str, List[dict]] = {}

    def _load_category(self, category: str) -> List[dict]:
        cat_dir = self.dataset_root / category
        if not cat_dir.exists():
            raise FileNotFoundError(f"Category '{category}' not found at {cat_dir}")

        obj_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()])
        examples = []
        for obj_dir in obj_dirs:
            if len(examples) >= self.max_per_cat:
                break
            ex = load_object(str(obj_dir), category)
            if ex is not None:
                examples.append(ex)
        return examples

    def get_examples(self, category: str, n: int = 5, seed: int = 0) -> List[dict]:
        """
        Return up to *n* random examples for *category*.

        Examples are loaded lazily and cached.
        """
        if category not in self._cache:
            self._cache[category] = self._load_category(category)

        pool = self._cache[category]
        rng = random.Random(seed)
        return rng.sample(pool, min(n, len(pool)))

    def format_examples_for_prompt(
        self,
        category: str,
        n: int = 5,
        seed: int = 0,
    ) -> str:
        """
        Return a compact string encoding of example objects suitable for
        embedding in an LLM prompt within the 32K token budget.

        Example output (one object per JSON block):
          Example 1:
          {"category":"chair","parts":[{"label":"seat","bbox_min":[-0.4,-0.05,-0.4],"bbox_max":[0.4,0.05,0.4]},…]}

          Example 2:
          …
        """
        examples = self.get_examples(category, n=n, seed=seed)
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(json.dumps(ex, separators=(",", ":")))
            lines.append("")
        return "\n".join(lines)

    def available_categories(self) -> List[str]:
        """Return sorted list of category names found in dataset_root."""
        return sorted([d.name for d in self.dataset_root.iterdir() if d.is_dir()])
