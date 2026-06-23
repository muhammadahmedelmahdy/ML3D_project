from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from project6_qwen_pipeline.models import (
    RawJointSource,
    RawMeshSource,
    RawObjectSource,
    RawPartSource,
)
from project6_qwen_pipeline.raw_geometry import (
    load_object_part_vertices,
    load_part_link_vertices,
)


class RawGeometryTests(unittest.TestCase):
    def test_combines_meshes_and_applies_visual_origins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.ply"
            second_path = root / "second.ply"
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(first_path)
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(second_path)

            part = RawPartSource(
                link_name="link_0",
                label="door",
                semantic_joint_type="hinge",
                parent_link="body",
                meshes=(
                    RawMeshSource(
                        first_path,
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0),
                    ),
                    RawMeshSource(
                        second_path,
                        (2.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0),
                    ),
                ),
            )

            vertices = load_part_link_vertices(part)
            np.testing.assert_allclose(vertices.min(axis=0), [-0.5, -0.5, -0.5])
            np.testing.assert_allclose(vertices.max(axis=0), [2.5, 0.5, 0.5])

    def test_places_nested_link_in_root_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh_path = Path(directory) / "door.ply"
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)
            door = RawPartSource(
                link_name="door",
                label="door",
                semantic_joint_type="hinge",
                parent_link="body",
                meshes=(
                    RawMeshSource(
                        mesh_path,
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0),
                    ),
                ),
            )
            raw_object = RawObjectSource(
                object_id="test",
                category="Cabinet",
                parts=(door,),
                joints=(
                    RawJointSource(
                        "body_joint",
                        "fixed",
                        "base",
                        "body",
                        (1.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                    RawJointSource(
                        "door_joint",
                        "revolute",
                        "body",
                        "door",
                        (0.0, 2.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                ),
            )

            vertices = load_object_part_vertices(raw_object)["door"]
            np.testing.assert_allclose(vertices.min(axis=0), [0.5, 1.5, -0.5])
            np.testing.assert_allclose(vertices.max(axis=0), [1.5, 2.5, 0.5])


if __name__ == "__main__":
    unittest.main()
