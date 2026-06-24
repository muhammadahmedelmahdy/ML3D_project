from __future__ import annotations

import json
from typing import Any, Sequence

from .extractor import validate_layout


MAX_CONTEXT_EXAMPLES = 50


def _compact_example(record: dict[str, Any]) -> dict[str, Any]:
    validate_layout(record)
    return {
        "category": record["category"],
        "parts": [
            {
                "id": part["id"],
                "label": part["label"],
                "bbox": part["bbox"],
            }
            for part in record["parts"]
        ],
    }


def build_layout_prompt(
    user_request: str,
    target_category: str,
    examples: Sequence[dict[str, Any]],
) -> str:
    """Build a compact prompt asking Qwen for one new labeled box layout."""
    user_request = user_request.strip()
    target_category = target_category.strip()
    if not user_request:
        raise ValueError("User request must not be empty")
    if not target_category:
        raise ValueError("Target category must not be empty")

    selected_examples = list(examples[:MAX_CONTEXT_EXAMPLES])
    compact_examples = [_compact_example(record) for record in selected_examples]
    examples_json = json.dumps(compact_examples, separators=(",", ":"))

    output_shape = {
        "category": target_category,
        "description": "short description of the proposed object",
        "parts": [
            {
                "id": "unique_part_instance_id",
                "label": "semantic_part_label",
                "bbox": {
                    "min": [0.0, 0.0, 0.0],
                    "max": [0.0, 0.0, 0.0],
                },
            }
        ],
    }
    output_json = json.dumps(output_shape, separators=(",", ":"))

    return "\n".join(
        [
            "You design coarse 3D object layouts from labeled axis-aligned bounding boxes.",
            "Coordinates are Z-up and every value must be inside [-0.5, 0.5].",
            "Each bbox uses opposite corners named min and max, with min <= max on every axis.",
            "Use the examples as layout references, but propose a new object rather than copying one.",
            f"Reference examples: {examples_json}",
            f"Target category: {target_category}",
            f"User request: {user_request}",
            "Return exactly one JSON object and no Markdown or additional text.",
            f"Required JSON shape: {output_json}",
        ]
    )

