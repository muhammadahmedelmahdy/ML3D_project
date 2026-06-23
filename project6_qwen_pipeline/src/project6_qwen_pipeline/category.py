from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


DEFAULT_CATEGORY_CATALOG = (
    Path(__file__).resolve().parent
    / "resources"
    / "partnet_mobility_categories.json"
)


def _canonical_categories(categories: Sequence[str]) -> tuple[str, ...]:
    canonical: list[str] = []
    seen: set[str] = set()

    for category in categories:
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Every category must be a non-empty string")
        clean_category = category.strip()
        key = clean_category.casefold()
        if key in seen:
            raise ValueError(f"Duplicate category: {clean_category}")
        seen.add(key)
        canonical.append(clean_category)

    if not canonical:
        raise ValueError("Category catalog must not be empty")
    return tuple(canonical)


def load_category_catalog(path: Path | None = None) -> tuple[str, ...]:
    """Load the allowed PartNet-Mobility categories."""
    catalog_path = Path(path) if path is not None else DEFAULT_CATEGORY_CATALOG
    with catalog_path.open("r", encoding="utf-8") as file:
        categories = json.load(file)
    if not isinstance(categories, list):
        raise ValueError("Category catalog must be a JSON list")
    return _canonical_categories(categories)


def build_category_prompt(
    user_request: str,
    categories: Sequence[str] | None = None,
) -> str:
    """Ask Qwen to map a natural-language request to one allowed category."""
    user_request = user_request.strip()
    if not user_request:
        raise ValueError("User request must not be empty")

    source_categories = load_category_catalog() if categories is None else categories
    allowed = _canonical_categories(source_categories)
    categories_json = json.dumps(allowed, separators=(",", ":"))
    return "\n".join(
        [
            "Classify the requested object into exactly one allowed PartNet-Mobility category.",
            "Choose the closest semantic category even if the user does not use its exact name.",
            f"Allowed categories: {categories_json}",
            f"User request: {user_request}",
            "Return exactly one JSON object and no Markdown or additional text.",
            'Required JSON shape: {"category":"one_allowed_category"}',
        ]
    )


def parse_category_response(
    response_text: str,
    categories: Sequence[str] | None = None,
) -> str:
    """Validate Qwen's category JSON and return the canonical category name."""
    source_categories = load_category_catalog() if categories is None else categories
    allowed = _canonical_categories(source_categories)
    allowed_by_key = {category.casefold(): category for category in allowed}

    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Qwen category response is not valid JSON: {error.msg}") from error

    if not isinstance(response, dict) or set(response) != {"category"}:
        raise ValueError("Qwen category response must contain only the category field")

    category = response["category"]
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Qwen category must be a non-empty string")

    canonical = allowed_by_key.get(category.strip().casefold())
    if canonical is None:
        raise ValueError(f"Qwen returned an unknown category: {category!r}")
    return canonical
