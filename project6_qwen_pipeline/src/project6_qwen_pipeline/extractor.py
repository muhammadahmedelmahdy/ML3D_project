from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .models import ObjectSource
from .raw_like_adapter import list_object_directories, load_object


LAYOUT_SCHEMA = "part-layout.v1"
PARTNET_Y_UP_TO_TRELLIS_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _load_vertices(mesh_path: Path) -> np.ndarray:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"Mesh is empty or unsupported: {mesh_path}")

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Mesh has invalid vertices: {mesh_path}")
    return vertices


def _rounded_vector(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values]


def build_layout_record(
    object_id: str,
    category: str,
    part_geometry: list[tuple[str, str, np.ndarray]],
) -> dict[str, Any]:
    """Normalize labeled part vertices and create the canonical layout record."""
    if not part_geometry:
        raise ValueError(f"Object {object_id!r} has no part geometry")

    for part_id, _label, vertices in part_geometry:
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(f"Part {part_id!r} has invalid vertices")
        if not np.isfinite(vertices).all():
            raise ValueError(f"Part {part_id!r} has non-finite vertices")

    all_vertices = np.concatenate(
        [vertices for _part_id, _label, vertices in part_geometry], axis=0
    )
    object_min = all_vertices.min(axis=0)
    object_max = all_vertices.max(axis=0)
    center = (object_min + object_max) / 2.0
    scale = float((object_max - object_min).max())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Object {object_id!r} has degenerate geometry")

    output_parts: list[dict[str, Any]] = []
    for part_id, label, vertices in part_geometry:
        normalized = (vertices - center) / scale
        output_parts.append(
            {
                "id": part_id,
                "label": label,
                "bbox": {
                    "min": _rounded_vector(normalized.min(axis=0)),
                    "max": _rounded_vector(normalized.max(axis=0)),
                },
            }
        )

    record: dict[str, Any] = {
        "schema_version": LAYOUT_SCHEMA,
        "object_id": object_id,
        "category": category,
        "coordinate_system": {
            "up_axis": "z",
            "range": [-0.5, 0.5],
            "normalization": "centered_uniform_object_bounds",
        },
        "parts": output_parts,
    }
    validate_layout(record)
    return record


def extract_object(source: ObjectSource) -> dict[str, Any]:
    """Compute normalized numerical boxes from prepared tutor part geometry."""
    part_geometry: list[tuple[str, str, np.ndarray]] = []
    for part in source.parts:
        vertices = _load_vertices(part.mesh_path)
        vertices_z_up = vertices @ PARTNET_Y_UP_TO_TRELLIS_Z_UP
        part_geometry.append((part.part_id, part.label, vertices_z_up))

    return build_layout_record(source.object_id, source.category, part_geometry)


def validate_layout(record: dict[str, Any], tolerance: float = 1e-6) -> None:
    """Validate the canonical layout record before it is used as Qwen context."""
    if record.get("schema_version") != LAYOUT_SCHEMA:
        raise ValueError("Unsupported layout schema")
    if not str(record.get("object_id", "")).strip():
        raise ValueError("Layout is missing an object ID")
    if not str(record.get("category", "")).strip():
        raise ValueError("Layout is missing a category")

    parts = record.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Layout contains no parts")

    seen_ids: set[str] = set()
    for part in parts:
        part_id = str(part.get("id", "")).strip()
        label = str(part.get("label", "")).strip()
        if not part_id or not label:
            raise ValueError("Every part needs a non-empty ID and label")
        if part_id in seen_ids:
            raise ValueError(f"Duplicate part ID: {part_id}")
        seen_ids.add(part_id)

        bbox = part.get("bbox", {})
        minimum = np.asarray(bbox.get("min", []), dtype=np.float64)
        maximum = np.asarray(bbox.get("max", []), dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError(f"Part {part_id!r} does not have 3D box corners")
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError(f"Part {part_id!r} has non-finite coordinates")
        if np.any(minimum > maximum):
            raise ValueError(f"Part {part_id!r} has min coordinates above max")
        if np.any(minimum < -0.5 - tolerance) or np.any(maximum > 0.5 + tolerance):
            raise ValueError(f"Part {part_id!r} lies outside the normalized cube")


def extract_dataset(dataset_root: Path, output_path: Path) -> int:
    """Extract all prepared objects and write one compact JSON record per line."""
    object_directories = list_object_directories(Path(dataset_root))
    if not object_directories:
        raise ValueError(f"No prepared objects found in {dataset_root}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for object_directory in object_directories:
            record = extract_object(load_object(object_directory))
            file.write(json.dumps(record, separators=(",", ":")))
            file.write("\n")

    return len(object_directories)
