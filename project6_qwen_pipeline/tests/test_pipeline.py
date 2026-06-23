from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from project6_qwen_pipeline.extractor import extract_dataset, extract_object
from project6_qwen_pipeline.models import ObjectSource, PartSource
from project6_qwen_pipeline.prepare_subset import prepare_tutor_subset
from project6_qwen_pipeline.raw_like_adapter import load_object


def export_box(
    path: Path,
    extents: tuple[float, float, float],
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = trimesh.transformations.translation_matrix(translation)
    trimesh.creation.box(extents=extents, transform=transform).export(path)


class LayoutPipelineTests(unittest.TestCase):
    def test_extracts_rotated_and_normalized_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh_path = Path(directory) / "part.ply"
            export_box(mesh_path, (2.0, 4.0, 6.0))
            source = ObjectSource(
                "object-1",
                "TestObject",
                (PartSource("part-1", "body", mesh_path),),
            )

            record = extract_object(source)
            bbox = record["parts"][0]["bbox"]

            np.testing.assert_allclose(bbox["min"], [-1 / 6, -0.5, -1 / 3], atol=1e-6)
            np.testing.assert_allclose(bbox["max"], [1 / 6, 0.5, 1 / 3], atol=1e-6)

    def test_duplicate_labels_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "wheel-left.ply"
            second = root / "wheel-right.ply"
            export_box(first, (1.0, 1.0, 1.0))
            export_box(second, (1.0, 1.0, 1.0))

            source = ObjectSource(
                "cart-1",
                "Cart",
                (
                    PartSource("wheel-left", "wheel", first),
                    PartSource("wheel-right", "wheel", second),
                ),
            )
            record = extract_object(source)
            self.assertEqual([part["label"] for part in record["parts"]], ["wheel", "wheel"])

    def test_parts_share_one_object_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.ply"
            right = root / "right.ply"
            export_box(left, (1.0, 1.0, 1.0), (-1.0, 0.0, 0.0))
            export_box(right, (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))

            source = ObjectSource(
                "object-2",
                "TestObject",
                (
                    PartSource("left", "side", left),
                    PartSource("right", "side", right),
                ),
            )
            parts = extract_object(source)["parts"]

            np.testing.assert_allclose(parts[0]["bbox"]["min"], [-0.5, -1 / 6, -1 / 6], atol=1e-6)
            np.testing.assert_allclose(parts[1]["bbox"]["max"], [0.5, 1 / 6, 1 / 6], atol=1e-6)

    def test_preparation_does_not_copy_provided_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tutor"
            object_directory = source / "100_Test"
            export_box(object_directory / "parts_orig" / "1_body.ply", (1.0, 2.0, 3.0))
            export_box(object_directory / "parts_bbox" / "1_body.ply", (9.0, 9.0, 9.0))
            manifest = {
                "obj_id": "100",
                "model_cat": "Test",
                "parts": [{"id": "1", "name": "body", "slug": "1_body"}],
            }
            with (object_directory / "parts_manifest.json").open("w", encoding="utf-8") as file:
                json.dump(manifest, file)

            prepared = root / "prepared"
            count = prepare_tutor_subset(source, prepared)

            self.assertEqual(count, 1)
            self.assertFalse(any(path.name == "parts_bbox" for path in prepared.rglob("*")))
            prepared_object = prepared / "100_Test"
            loaded = load_object(prepared_object)
            self.assertEqual(loaded.parts[0].label, "body")

            output = root / "layouts.jsonl"
            self.assertEqual(extract_dataset(prepared, output), 1)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
