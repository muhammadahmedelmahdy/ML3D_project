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

from project6_qwen_pipeline.raw_layout import extract_raw_partnet_object


class RawLayoutTests(unittest.TestCase):
    def test_extracts_normalized_box_from_raw_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "42"
            mesh_path = root / "textured_objs" / "body.obj"
            mesh_path.parent.mkdir(parents=True)
            trimesh.creation.box(extents=(2.0, 4.0, 6.0)).export(mesh_path)

            (root / "meta.json").write_text(
                json.dumps({"anno_id": "42", "model_cat": "Box"}),
                encoding="utf-8",
            )
            (root / "semantics.txt").write_text(
                "link_0 static body\n",
                encoding="utf-8",
            )
            (root / "mobility.urdf").write_text(
                """<robot name="test">
  <link name="base"/>
  <link name="link_0">
    <visual><geometry><mesh filename="textured_objs/body.obj"/></geometry></visual>
  </link>
  <joint name="joint_0" type="fixed">
    <parent link="base"/><child link="link_0"/><origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
</robot>
""",
                encoding="utf-8",
            )

            record = extract_raw_partnet_object(root)

            self.assertEqual(record["object_id"], "42")
            self.assertEqual(record["category"], "Box")
            self.assertEqual(record["parts"][0]["label"], "body")
            np.testing.assert_allclose(
                record["parts"][0]["bbox"]["min"],
                [-1 / 6, -1 / 3, -0.5],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                record["parts"][0]["bbox"]["max"],
                [1 / 6, 1 / 3, 0.5],
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
