from __future__ import annotations

import json
from pathlib import Path

from .models import ObjectSource, PartSource


METADATA_FILENAME = "metadata.json"
RAW_LIKE_SCHEMA = "partnet-raw-like.v1"


def list_object_directories(dataset_root: Path) -> list[Path]:
    """Return prepared object directories in stable order."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / METADATA_FILENAME).is_file()
    )


def load_object(object_directory: Path) -> ObjectSource:
    """Load one object from the stripped, raw-like subset format."""
    object_directory = Path(object_directory)
    metadata_path = object_directory / METADATA_FILENAME

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("schema_version") != RAW_LIKE_SCHEMA:
        raise ValueError(f"Unsupported metadata schema in {metadata_path}")

    object_id = str(metadata.get("object_id", "")).strip()
    category = str(metadata.get("category", "")).strip()
    if not object_id or not category:
        raise ValueError(f"Missing object ID or category in {metadata_path}")

    part_sources: list[PartSource] = []
    seen_ids: set[str] = set()

    for part in metadata.get("parts", []):
        part_id = str(part.get("id", "")).strip()
        label = str(part.get("label", "")).strip()
        relative_mesh_path = Path(str(part.get("mesh", "")))
        mesh_path = object_directory / relative_mesh_path

        if not part_id or not label:
            raise ValueError(f"Part with missing ID or label in {metadata_path}")
        if part_id in seen_ids:
            raise ValueError(f"Duplicate part ID {part_id!r} in {metadata_path}")
        if relative_mesh_path.is_absolute() or ".." in relative_mesh_path.parts:
            raise ValueError(f"Unsafe mesh path for part {part_id!r}")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing mesh for part {part_id!r}: {mesh_path}")

        seen_ids.add(part_id)
        part_sources.append(PartSource(part_id, label, mesh_path))

    if not part_sources:
        raise ValueError(f"No parts listed in {metadata_path}")

    return ObjectSource(object_id, category, tuple(part_sources))

