from __future__ import annotations

import argparse
import json
from pathlib import Path

from .category import build_category_prompt, parse_category_response
from .extractor import extract_dataset
from .prepare_subset import prepare_tutor_subset
from .prompting import build_layout_prompt
from .qwen_runner import DEFAULT_MODEL_PATH, run_qwen_file
from .raw_layout import write_raw_partnet_layout
from .response import parse_layout_response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare PartNet data and extract normalized part layouts."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="Strip the tutor subset into a raw-like dataset."
    )
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    extract = commands.add_parser(
        "extract", help="Extract normalized labeled boxes into JSONL."
    )
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)

    extract_raw = commands.add_parser(
        "extract-raw", help="Extract one raw PartNet-Mobility object into JSON."
    )
    extract_raw.add_argument("--input", type=Path, required=True)
    extract_raw.add_argument("--output", type=Path, required=True)

    prompt = commands.add_parser(
        "build-prompt", help="Build a Qwen prompt from existing layout records."
    )
    prompt.add_argument("--request", required=True)
    prompt.add_argument("--category", required=True)
    prompt.add_argument("--examples", type=Path, nargs="+", required=True)
    prompt.add_argument("--output", type=Path, required=True)

    validate_response = commands.add_parser(
        "validate-response", help="Validate a Qwen layout response file."
    )
    validate_response.add_argument("--input", type=Path, required=True)
    validate_response.add_argument("--category")

    category_prompt = commands.add_parser(
        "build-category-prompt",
        help="Build the first Qwen prompt from only a user request.",
    )
    category_prompt.add_argument("--request", required=True)
    category_prompt.add_argument("--output", type=Path, required=True)

    validate_category = commands.add_parser(
        "validate-category-response",
        help="Validate Qwen's category-classification response.",
    )
    validate_category.add_argument("--input", type=Path, required=True)

    run_qwen = commands.add_parser(
        "run-qwen",
        help="Run one prompt file through local Qwen and save the raw answer.",
    )
    run_qwen.add_argument("--prompt", type=Path, required=True)
    run_qwen.add_argument("--output", type=Path, required=True)
    run_qwen.add_argument("--model", default=DEFAULT_MODEL_PATH)
    run_qwen.add_argument("--max-new-tokens", type=int, default=2048)
    run_qwen.add_argument("--temperature", type=float, default=0.7)
    run_qwen.add_argument("--top-p", type=float, default=0.8)
    run_qwen.add_argument("--top-k", type=int, default=20)
    run_qwen.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode. Keep this off for strict JSON tasks.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "prepare":
        count = prepare_tutor_subset(args.source, args.output)
        print(f"Prepared {count} objects in {args.output}")
        return

    if args.command == "extract-raw":
        write_raw_partnet_layout(args.input, args.output)
        print(f"Extracted raw object layout to {args.output}")
        return

    if args.command == "build-prompt":
        examples = []
        for example_path in args.examples:
            with example_path.open("r", encoding="utf-8") as file:
                examples.append(json.load(file))
        prompt_text = build_layout_prompt(args.request, args.category, examples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt_text + "\n", encoding="utf-8")
        print(f"Built Qwen prompt with {min(len(examples), 5)} examples at {args.output}")
        return

    if args.command == "validate-response":
        response_text = args.input.read_text(encoding="utf-8")
        response = parse_layout_response(
            response_text,
            expected_category=args.category,
        )
        print(f"Valid {response['category']} layout with {len(response['parts'])} parts")
        return

    if args.command == "build-category-prompt":
        prompt_text = build_category_prompt(args.request)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt_text + "\n", encoding="utf-8")
        print(f"Built category prompt at {args.output}")
        return

    if args.command == "validate-category-response":
        category = parse_category_response(args.input.read_text(encoding="utf-8"))
        print(f"Valid PartNet-Mobility category: {category}")
        return

    if args.command == "run-qwen":
        response_text = run_qwen_file(
            args.prompt,
            args.output,
            model_path=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            thinking=args.thinking,
        )
        print(f"Wrote Qwen response to {args.output}")
        print(response_text)
        return

    count = extract_dataset(args.input, args.output)
    print(f"Extracted {count} object layouts to {args.output}")


if __name__ == "__main__":
    main()
