from __future__ import annotations

from pathlib import Path


DEFAULT_MODEL_PATH = "/cluster/52/jonasclotten/shared/project6/models/Qwen3-8B"


def run_qwen_prompt(
    prompt_text: str,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    thinking: bool = False,
) -> str:
    """Load Qwen and return its answer for one prompt.

    This is meant to run on the VM, where the model weights live. It is not an
    HTTP server and it does not call an external API.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    clean_prompt = prompt_text.strip()
    if not clean_prompt:
        raise ValueError("Prompt text must not be empty")

    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "Return only the requested machine-readable output."},
        {"role": "user", "content": clean_prompt},
    ]

    chat_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        text = tokenizer.apply_chat_template(
            messages,
            enable_thinking=thinking,
            **chat_kwargs,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **chat_kwargs)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    answer_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]
    return _strip_qwen_thinking(response_text).strip()


def run_qwen_file(
    prompt_path: Path,
    output_path: Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    thinking: bool = False,
) -> str:
    """Read one prompt file, run Qwen, and write the raw response file."""
    prompt_text = prompt_path.read_text(encoding="utf-8")
    response_text = run_qwen_prompt(
        prompt_text,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        thinking=thinking,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response_text + "\n", encoding="utf-8")
    return response_text


def _strip_qwen_thinking(response_text: str) -> str:
    """Remove Qwen3 thinking text when it appears before the final answer."""
    marker = "</think>"
    if marker not in response_text:
        return response_text
    return response_text.split(marker, maxsplit=1)[1]
