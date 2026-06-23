from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from project6_qwen_pipeline.partnet_raw_adapter import load_raw_partnet_object


class PartNetRawAdapterTests(unittest.TestCase):
    def test_maps_public_sample_semantics_to_urdf_links(self) -> None:
        fixture = PROJECT_ROOT / "tests" / "fixtures" / "partnet_raw" / "7179"
        raw_object = load_raw_partnet_object(fixture, require_mesh_files=False)

        self.assertEqual(raw_object.object_id, "7179")
        self.assertEqual(raw_object.category, "Oven")
        self.assertEqual(len(raw_object.parts), 6)
        self.assertEqual(len(raw_object.joints), 6)

        first_part = raw_object.parts[0]
        self.assertEqual(first_part.link_name, "link_0")
        self.assertEqual(first_part.label, "translation_tray")
        self.assertEqual(first_part.semantic_joint_type, "slider")
        self.assertEqual(first_part.parent_link, "link_5")
        self.assertEqual(len(first_part.mesh_paths), 18)
        self.assertEqual(first_part.meshes[0].origin_xyz, (0.0, 0.0, 0.0))
        self.assertTrue(
            all(path.parts[-2] == "textured_objs" for path in first_part.mesh_paths)
        )

        first_door = raw_object.parts[3]
        self.assertEqual(first_door.label, "door")
        self.assertEqual(
            first_door.meshes[0].origin_xyz,
            (0.5342130064964294, 0.7690419264409818, -0.36910544487147934),
        )

        self.assertEqual(raw_object.parts[-1].label, "oven_body")
        self.assertEqual(raw_object.parts[-1].parent_link, "base")

        fixed_joint = raw_object.joints[-1]
        self.assertEqual(fixed_joint.parent_link, "base")
        self.assertEqual(fixed_joint.child_link, "link_5")
        self.assertEqual(fixed_joint.joint_type, "fixed")


if __name__ == "__main__":
    unittest.main()
