"""All held-out prompts, identical decoding, explicit heuristic scoring."""

import argparse
import contextlib
import hashlib
import json
import re
from pathlib import Path
import torch
from peft import PeftModel
from .data import ROOT, read_jsonl
from .model import load, generate


def score(answer, required):
    text = answer.lower()
    return all(
        re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", text)
        for term in required
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=ROOT / "outputs/lora")
    p.add_argument("--output", type=Path, default=ROOT / "reports/results.json")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    for split, expected in manifest["data_sha256"].items():
        if (
            hashlib.sha256((ROOT / f"data/{split}.jsonl").read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError(f"Data changed after training: {split}")
    torch.set_num_threads(4)
    base, tok = load(cfg, args.device)
    model = PeftModel.from_pretrained(base, args.run / "adapter")
    cases = read_jsonl(ROOT / "data/test.jsonl")
    for mode in ["before", "after"]:
        with model.disable_adapter() if mode == "before" else contextlib.nullcontext():
            for case in cases:
                answer = generate(model, tok, case["question"], cfg["max_new_tokens"])
                case[mode] = answer
                case[mode + "_pass"] = bool(score(answer, case["required"]))
                print(f"{mode} {case['id']}: {answer}", flush=True)
    summary = {
        category: {
            mode: sum(c[mode + "_pass"] for c in cases if c["category"] == category)
            / sum(c["category"] == category for c in cases)
            for mode in ["before", "after"]
        }
        for category in ["known", "unknown"]
    }
    result = dict(
        summary=summary,
        cases=cases,
        manifest=manifest,
        decoding=dict(do_sample=False, max_new_tokens=cfg["max_new_tokens"]),
        scoring="All required phrases must appear with word boundaries. This is a heuristic, not a semantic correctness judge.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
