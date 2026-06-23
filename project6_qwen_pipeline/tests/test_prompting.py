from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from project6_qwen_pipeline.prompting import build_layout_prompt


def layout(object_id: str) -> dict:
    return {
        "schema_version": "part-layout.v1",
        "object_id": object_id,
        "category": "Oven",
        "coordinate_system": {
            "up_axis": "z",
            "range": [-0.5, 0.5],
            "normalization": "centered_uniform_object_bounds",
        },
        "parts": [
            {
                "id": "body",
                "label": "oven_body",
                "bbox": {
                    "min": [-0.5, -0.4, -0.3],
                    "max": [0.5, 0.4, 0.3],
                },
            }
        ],
    }


class PromptingTests(unittest.TestCase):
    def test_builds_prompt_with_one_example(self) -> None:
        prompt = build_layout_prompt(
            "An oven with two front doors.",
            "Oven",
            [layout("7179")],
        )

        self.assertIn("Target category: Oven", prompt)
        self.assertIn("An oven with two front doors.", prompt)
        self.assertIn('"label":"oven_body"', prompt)
        self.assertIn("Return exactly one JSON object", prompt)

    def test_uses_at_most_five_examples(self) -> None:
        examples = [layout(str(index)) for index in range(7)]
        prompt = build_layout_prompt("A compact oven.", "Oven", examples)
        reference_line = next(
            line for line in prompt.splitlines() if line.startswith("Reference examples: ")
        )
        encoded_examples = reference_line.removeprefix("Reference examples: ")

        self.assertEqual(len(json.loads(encoded_examples)), 5)

    def test_rejects_invalid_example(self) -> None:
        invalid = layout("bad")
        invalid["parts"][0]["bbox"]["max"][0] = 0.6

        with self.assertRaisesRegex(ValueError, "outside the normalized cube"):
            build_layout_prompt("An oven.", "Oven", [invalid])


if __name__ == "__main__":
    unittest.main()

