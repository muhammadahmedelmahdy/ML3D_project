from __future__ import annotations

import json
import math
from typing import Any


TOP_LEVEL_FIELDS = {"category", "description", "parts"}
PART_FIELDS = {"id", "label", "bbox"}
BBOX_FIELDS = {"min", "max"}


def _require_exact_fields(value: dict[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Invalid fields at {location}: missing={missing}, extra={extra}")


def _coordinate_vector(value: Any, location: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{location} must contain exactly three coordinates")

    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(f"{location} coordinates must be numbers")
        number = float(coordinate)
        if not math.isfinite(number):
            raise ValueError(f"{location} coordinates must be finite")
        if number < -0.5 or number > 0.5:
            raise ValueError(f"{location} coordinates must be inside [-0.5, 0.5]")
        coordinates.append(number)
    return coordinates


def parse_layout_response(
    response_text: str,
    *,
    expected_category: str | None = None,
) -> dict[str, Any]:
    """Parse and strictly validate Qwen's proposed layout JSON."""
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Qwen response is not valid JSON: {error.msg}") from error

    if not isinstance(response, dict):
        raise ValueError("Qwen response must be one JSON object")
    _require_exact_fields(response, TOP_LEVEL_FIELDS, "root")

    category = response["category"]
    description = response["description"]
    parts = response["parts"]
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-empty string")
    if expected_category is not None and category.casefold() != expected_category.casefold():
        raise ValueError(
            f"Expected category {expected_category!r}, received {category!r}"
        )
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    if not isinstance(parts, list) or not parts:
        raise ValueError("parts must be a non-empty list")

    seen_ids: set[str] = set()
    for index, part in enumerate(parts):
        location = f"parts[{index}]"
        if not isinstance(part, dict):
            raise ValueError(f"{location} must be an object")
        _require_exact_fields(part, PART_FIELDS, location)

        part_id = part["id"]
        label = part["label"]
        bbox = part["bbox"]
        if not isinstance(part_id, str) or not part_id.strip():
            raise ValueError(f"{location}.id must be a non-empty string")
        if part_id in seen_ids:
            raise ValueError(f"Duplicate part ID: {part_id}")
        seen_ids.add(part_id)
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{location}.label must be a non-empty string")
        if not isinstance(bbox, dict):
            raise ValueError(f"{location}.bbox must be an object")
        _require_exact_fields(bbox, BBOX_FIELDS, f"{location}.bbox")

        minimum = _coordinate_vector(bbox["min"], f"{location}.bbox.min")
        maximum = _coordinate_vector(bbox["max"], f"{location}.bbox.max")
        if any(low > high for low, high in zip(minimum, maximum)):
            raise ValueError(f"{location}.bbox has min coordinates above max")

    return response

