from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from project6_qwen_pipeline.category import (
    build_category_prompt,
    load_category_catalog,
    parse_category_response,
)


class CategoryTests(unittest.TestCase):
    def test_default_catalog_contains_expected_categories(self) -> None:
        categories = load_category_catalog()
        self.assertEqual(len(categories), 46)
        self.assertIn("Chair", categories)
        self.assertIn("Oven", categories)

    def test_builds_prompt_from_only_user_request(self) -> None:
        prompt = build_category_prompt("Create a chair with three legs")
        self.assertIn("Create a chair with three legs", prompt)
        self.assertIn('"Chair"', prompt)
        self.assertNotIn("bbox", prompt.casefold())

    def test_accepts_known_category_and_returns_canonical_case(self) -> None:
        category = parse_category_response('{"category":"chair"}')
        self.assertEqual(category, "Chair")

    def test_rejects_unknown_category(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown category"):
            parse_category_response('{"category":"Sofa"}')

    def test_rejects_extra_response_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "only the category field"):
            parse_category_response('{"category":"Chair","reason":"sitting"}')


if __name__ == "__main__":
    unittest.main()

