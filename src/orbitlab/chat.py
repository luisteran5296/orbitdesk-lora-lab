"""Print a matched before/after response for a new question."""

import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from .data import ROOT
from .model import load, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/lora")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not args.question.strip() or len(args.question) > 1200:
        parser.error("Use a nonempty question of at most 1,200 characters.")
    torch.set_num_threads(4)
    cfg = json.loads((args.run / "config.json").read_text())
    base, tok = load(cfg, args.device)
    model = PeftModel.from_pretrained(base, args.run / "adapter")
    with model.disable_adapter():
        before = generate(model, tok, args.question, cfg["max_new_tokens"])
    after = generate(model, tok, args.question, cfg["max_new_tokens"])
    print(f"QUESTION\n{args.question}\n\nBEFORE\n{before}\n\nAFTER\n{after}")


if __name__ == "__main__":
    main()
