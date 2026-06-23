from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .extractor import build_layout_record
from .partnet_raw_adapter import load_raw_partnet_object
from .raw_geometry import load_object_part_vertices


def extract_raw_partnet_object(object_directory: Path) -> dict[str, Any]:
    """Convert one raw PartNet-Mobility object into normalized labeled boxes."""
    raw_object = load_raw_partnet_object(Path(object_directory))
    vertices_by_link = load_object_part_vertices(raw_object)
    part_geometry = [
        (part.link_name, part.label, vertices_by_link[part.link_name])
        for part in raw_object.parts
    ]
    return build_layout_record(
        raw_object.object_id,
        raw_object.category,
        part_geometry,
    )


def write_raw_partnet_layout(object_directory: Path, output_path: Path) -> None:
    """Extract one raw object and write an indented JSON record."""
    record = extract_raw_partnet_object(object_directory)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
        file.write("\n")
