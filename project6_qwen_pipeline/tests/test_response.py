from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from project6_qwen_pipeline.response import parse_layout_response


def valid_response() -> dict:
    return {
        "category": "Oven",
        "description": "A compact oven with two front doors.",
        "parts": [
            {
                "id": "body_1",
                "label": "oven_body",
                "bbox": {
                    "min": [-0.4, -0.3, -0.5],
                    "max": [0.4, 0.3, 0.5],
                },
            },
            {
                "id": "door_1",
                "label": "door",
                "bbox": {
                    "min": [-0.41, -0.31, -0.3],
                    "max": [-0.35, 0.31, 0.3],
                },
            },
        ],
    }


class ResponseTests(unittest.TestCase):
    def test_accepts_valid_strict_json(self) -> None:
        parsed = parse_layout_response(
            json.dumps(valid_response()),
            expected_category="oven",
        )
        self.assertEqual(len(parsed["parts"]), 2)

    def test_rejects_markdown_code_fence(self) -> None:
        text = "```json\n" + json.dumps(valid_response()) + "\n```"
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            parse_layout_response(text)

    def test_rejects_duplicate_part_ids(self) -> None:
        response = valid_response()
        response["parts"][1]["id"] = "body_1"
        with self.assertRaisesRegex(ValueError, "Duplicate part ID"):
            parse_layout_response(json.dumps(response))

    def test_rejects_out_of_range_coordinate(self) -> None:
        response = valid_response()
        response["parts"][0]["bbox"]["max"][2] = 0.6
        with self.assertRaisesRegex(ValueError, r"inside \[-0.5, 0.5\]"):
            parse_layout_response(json.dumps(response))

    def test_rejects_reversed_box_corners(self) -> None:
        response = valid_response()
        response["parts"][0]["bbox"]["min"][0] = 0.4
        response["parts"][0]["bbox"]["max"][0] = -0.4
        with self.assertRaisesRegex(ValueError, "min coordinates above max"):
            parse_layout_response(json.dumps(response))

    def test_rejects_unexpected_fields(self) -> None:
        response = valid_response()
        response["notes"] = "extra output"
        with self.assertRaisesRegex(ValueError, "Invalid fields at root"):
            parse_layout_response(json.dumps(response))


if __name__ == "__main__":
    unittest.main()
