"""
Qwen3-8B layout proposer.

Given a category and in-context examples from PartNet-Mobility, prompts
Qwen3-8B (thinking mode) to output a set of labeled part bounding boxes
and a short text description in strict JSON.

Expected JSON output schema:
  {
    "description": "<short text description for TRELLIS>",
    "parts": [
      {
        "label": "<part name>",
        "bbox_min": [x, y, z],   // in [-0.5, 0.5] normalized space
        "bbox_max": [x, y, z]
      },
      ...
    ]
  }
"""

from __future__ import annotations
import json
import re
from typing import List, Optional

import torch


SYSTEM_PROMPT = """\
You are a 3D object layout designer. Given a category and examples of real \
objects with labeled part bounding boxes, propose a new, plausible object of \
that category.

Rules:
- Output ONLY valid JSON — no prose before or after.
- All bounding box coordinates must be in the range [-0.5, 0.5] \
(normalized object space, Y is up).
- Parts must not overlap except at shared boundaries.
- Include a short "description" field (1-2 sentences) suitable as a text \
prompt for a 3D generative model.
- Think carefully inside <think>…</think> before writing the JSON.

Output schema (strictly):
{
  "description": "...",
  "parts": [
    {"label": "...", "bbox_min": [x,y,z], "bbox_max": [x,y,z]},
    ...
  ]
}
"""

USER_TEMPLATE = """\
Category: {category}

Here are {n} example objects from this category (coordinates in [-0.5, 0.5]):

{examples}

Now propose a new {category}. Output only the JSON object.\
"""


def _build_messages(category: str, examples_str: str, n: int) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                category=category,
                n=n,
                examples=examples_str,
            ),
        },
    ]


def _extract_json(text: str) -> Optional[dict]:
    """
    Extract and parse the first JSON object from the model output.
    Strips <think>…</think> blocks first.
    """
    # Remove thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _validate_layout(layout: dict) -> bool:
    """Basic sanity checks on the parsed layout."""
    if "description" not in layout or "parts" not in layout:
        return False
    if not isinstance(layout["parts"], list) or len(layout["parts"]) == 0:
        return False
    for p in layout["parts"]:
        if not all(k in p for k in ("label", "bbox_min", "bbox_max")):
            return False
        if len(p["bbox_min"]) != 3 or len(p["bbox_max"]) != 3:
            return False
        for v in p["bbox_min"] + p["bbox_max"]:
            if not isinstance(v, (int, float)):
                return False
    return True


class QwenLayoutProposer:
    """
    Wraps Qwen3-8B to propose part-level bounding box layouts for a given
    object category, conditioned on PartNet-Mobility in-context examples.

    Args:
        model_name: HuggingFace model ID (default "Qwen/Qwen3-8B").
        device: "cuda", "cpu", or "auto".
        max_new_tokens: Generation budget (includes thinking tokens).
        temperature: Sampling temperature.
        top_p: Nucleus sampling p.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "auto",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def propose(
        self,
        category: str,
        examples_str: str,
        n_examples: int,
        retries: int = 3,
    ) -> dict:
        """
        Generate a part layout for *category* given *examples_str*.

        Args:
            category: Object category (e.g. "chair", "table").
            examples_str: Pre-formatted context string from
                          PartNetMobilityContextBuilder.format_examples_for_prompt().
            n_examples: Number of examples in *examples_str* (used in prompt).
            retries: How many times to retry if JSON parsing/validation fails.

        Returns:
            Validated layout dict with keys "description" and "parts".

        Raises:
            RuntimeError: If all retries fail to produce a valid layout.
        """
        messages = _build_messages(category, examples_str, n_examples)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,   # Qwen3 thinking mode
        )

        for attempt in range(retries):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )
            # Decode only the newly generated tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            layout = _extract_json(response)
            if layout is not None and _validate_layout(layout):
                return layout

            print(f"[QwenLayoutProposer] Attempt {attempt+1}/{retries} failed. "
                  f"Raw output:\n{response[:500]}\n")

        raise RuntimeError(
            f"QwenLayoutProposer failed to produce a valid layout for "
            f"'{category}' after {retries} attempts."
        )
