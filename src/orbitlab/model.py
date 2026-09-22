"""Shared model loading ensures baseline and adapter use identical inference."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .data import SYSTEM


def load(config, device="auto"):
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    kwargs = dict(revision=config["revision"], dtype=dtype, attn_implementation="sdpa")
    if config["method"] == "qlora":
        if device != "cuda":
            raise ValueError("QLoRA requires CUDA in this project. Use LoRA for CPU.")
        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            ),
            device_map={"": 0},
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], **kwargs)
    if config["method"] != "qlora":
        model.to(device)
    return model, tokenizer


def messages(question):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]


@torch.inference_mode()
def generate(model, tokenizer, question, max_new_tokens=96):
    model.eval()
    inputs = tokenizer.apply_chat_template(
        messages(question),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def encode(row, tokenizer, max_length):
    prompt = tokenizer.apply_chat_template(
        messages(row["question"]), tokenize=True, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        messages(row["question"]) + [{"role": "assistant", "content": row["answer"]}],
        tokenize=True,
    )
    if full[: len(prompt)] != prompt:
        raise ValueError(
            "Chat template prefix mismatch; assistant-only masking is unsafe."
        )
    if len(full) > max_length:
        raise ValueError(
            "Example exceeds max_length; increase the limit rather than truncate the answer."
        )
    return {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt) :]}


def collate(rows, pad_id):
    length = max(len(r["input_ids"]) for r in rows)
    return {
        "input_ids": torch.tensor(
            [r["input_ids"] + [pad_id] * (length - len(r["input_ids"])) for r in rows]
        ),
        "labels": torch.tensor(
            [r["labels"] + [-100] * (length - len(r["labels"])) for r in rows]
        ),
        "attention_mask": torch.tensor(
            [
                [1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"]))
                for r in rows
            ]
        ),
    }
