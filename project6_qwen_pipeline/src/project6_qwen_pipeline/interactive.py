from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .category import build_category_prompt, parse_category_response
from .extractor import validate_layout
from .prompting import build_layout_prompt
from .qwen_runner import DEFAULT_MODEL_PATH, run_qwen_file
from .response import parse_layout_response


DEFAULT_PROMPT_DIR = Path("prompts")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_PROCESSED_DIR = Path("/cluster/52/jonasclotten/shared/project6/data/processed")


def main() -> None:
    request = input("What should Qwen design?\n> ").strip()
    if not request:
        raise SystemExit("No request entered.")

    result = run_interactive_pipeline(request)
    print("\nDone.")
    print(f"Category: {result['category']}")
    print(f"Category prompt: {result['category_prompt_path']}")
    print(f"Category output: {result['category_output_path']}")

    if result.get("layout_prompt_path") is None:
        print("No matching layout examples were found, so the layout step was skipped.")
        print("Prepare/extract PartNet examples first, then run this again.")
        return

    print(f"Layout prompt: {result['layout_prompt_path']}")
    print(f"Layout output: {result['layout_output_path']}")
    print(f"Layout parts: {result['layout_part_count']}")


def run_interactive_pipeline(
    user_request: str,
    *,
    prompt_dir: Path = DEFAULT_PROMPT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Run category selection and, when examples exist, layout generation."""
    clean_request = user_request.strip()
    if not clean_request:
        raise ValueError("User request must not be empty")

    slug = _slugify(clean_request)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_prompt_path = prompt_dir / f"category_{slug}.txt"
    category_output_path = output_dir / f"category_{slug}.json"

    category_prompt = build_category_prompt(clean_request)
    category_prompt_path.write_text(category_prompt + "\n", encoding="utf-8")
    run_qwen_file(category_prompt_path, category_output_path, model_path=model_path, thinking=False)
    category = parse_category_response(category_output_path.read_text(encoding="utf-8"))

    result: dict[str, Any] = {
        "category": category,
        "category_prompt_path": category_prompt_path,
        "category_output_path": category_output_path,
        "layout_prompt_path": None,
        "layout_output_path": None,
        "layout_part_count": None,
    }

    examples = load_layout_examples(processed_dir, category, limit=20)
    if not examples:
        return result

    layout_prompt_path = prompt_dir / f"layout_{slug}.txt"
    layout_output_path = output_dir / f"layout_{slug}.json"
    layout_prompt = build_layout_prompt(clean_request, category, examples)
    layout_prompt_path.write_text(layout_prompt + "\n", encoding="utf-8")
    run_qwen_file(layout_prompt_path, layout_output_path, model_path=model_path, thinking=False)

    layout = parse_layout_response(
        layout_output_path.read_text(encoding="utf-8"),
        expected_category=category,
    )
    result["layout_prompt_path"] = layout_prompt_path
    result["layout_output_path"] = layout_output_path
    result["layout_part_count"] = len(layout["parts"])
    return result


def load_layout_examples(
    processed_dir: Path,
    category: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Load up to `limit` prepared layout records for one category."""
    processed_dir = Path(processed_dir)
    if limit <= 0 or not processed_dir.exists():
        return []

    examples: list[dict[str, Any]] = []
    layouts_jsonl = processed_dir / "layouts.jsonl"
    if layouts_jsonl.exists():
        for line in layouts_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                _try_add_example(examples, json.loads(line), category, limit)
            if len(examples) >= limit:
                return examples

    for path in sorted(processed_dir.glob("*.json")):
        if path.name == "layouts.jsonl":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        _try_add_example(examples, record, category, limit)
        if len(examples) >= limit:
            break

    return examples


def _try_add_example(
    examples: list[dict[str, Any]],
    record: dict[str, Any],
    category: str,
    limit: int,
) -> None:
    if len(examples) >= limit:
        return
    if not isinstance(record, dict):
        return
    if str(record.get("category", "")).casefold() != category.casefold():
        return
    try:
        validate_layout(record)
    except ValueError:
        return
    examples.append(record)


def _slugify(text: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    if not slug:
        return "request"
    return slug[:max_length].strip("_") or "request"


if __name__ == "__main__":
    main()
