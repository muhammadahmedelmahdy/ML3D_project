from __future__ import annotations

import json
import shutil
from pathlib import Path

from .raw_like_adapter import RAW_LIKE_SCHEMA


def _manifest_parts(manifest: dict, manifest_path: Path) -> list[dict]:
    parts = manifest.get("parts", [])
    if not isinstance(parts, list) or not parts:
        raise ValueError(f"Manifest has no parts list: {manifest_path}")
    return parts


def prepare_tutor_subset(source_root: Path, output_root: Path) -> int:
    """Copy only original part meshes and equivalent raw-dataset metadata."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"Tutor subset does not exist: {source_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Output must not be inside the tutor subset source")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_root}")

    object_directories = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "parts_manifest.json").is_file()
    )
    if not object_directories:
        raise ValueError(f"No tutor subset objects found in {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    for object_directory in object_directories:
        manifest_path = object_directory / "parts_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        if not isinstance(manifest, dict):
            raise ValueError(f"Expected an object manifest in {manifest_path}")

        object_id = str(manifest.get("obj_id", object_directory.name)).strip()
        category = str(manifest.get("model_cat", "")).strip()
        if not object_id or not category:
            raise ValueError(f"Missing object ID or category in {manifest_path}")

        destination = output_root / object_directory.name
        destination_parts = destination / "parts"
        destination_parts.mkdir(parents=True, exist_ok=True)

        clean_parts: list[dict[str, str]] = []
        for part in _manifest_parts(manifest, manifest_path):
            slug = str(part.get("slug", "")).strip()
            label = str(part.get("name", "")).strip()
            part_id = str(part.get("id", slug)).strip()
            source_mesh = object_directory / "parts_orig" / f"{slug}.ply"

            if not slug or not label or not part_id:
                raise ValueError(f"Part with missing fields in {manifest_path}")
            if not source_mesh.is_file():
                raise FileNotFoundError(f"Missing original part mesh: {source_mesh}")

            destination_mesh = destination_parts / f"{slug}.ply"
            shutil.copy2(source_mesh, destination_mesh)
            clean_parts.append(
                {
                    "id": part_id,
                    "label": label,
                    "mesh": destination_mesh.relative_to(destination).as_posix(),
                }
            )

        clean_metadata = {
            "schema_version": RAW_LIKE_SCHEMA,
            "object_id": object_id,
            "category": category,
            "parts": clean_parts,
        }
        metadata_path = destination / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(clean_metadata, file, indent=2)
            file.write("\n")

    return len(object_directories)
