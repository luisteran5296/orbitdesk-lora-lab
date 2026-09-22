"""An intentionally explicit PyTorch training loop: no hidden SFT trainer."""

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import model_info
from .data import ROOT, read_jsonl
from .model import load, encode, collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lora.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lora")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if cfg["epochs"] < 1:
        raise ValueError("epochs must be positive")
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    # Record immutable source weights even when the human configuration says main.
    cfg["revision"] = model_info(cfg["model_id"], revision=cfg["revision"]).sha
    model, tok = load(cfg, args.device)
    if cfg["method"] == "qlora":
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["alpha"],
            lora_dropout=cfg["dropout"],
            target_modules=cfg["target_modules"],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()
    loaders = {}
    for split in ["train", "validation"]:
        rows = [
            encode(r, tok, cfg["max_length"])
            for r in read_jsonl(ROOT / f"data/{split}.jsonl")
        ]
        loaders[split] = DataLoader(
            rows,
            batch_size=cfg["batch_size"],
            shuffle=split == "train",
            collate_fn=partial(collate, pad_id=tok.pad_token_id),
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=0.01,
    )
    updates = (
        math.ceil(len(loaders["train"]) / cfg["gradient_accumulation"]) * cfg["epochs"]
    )
    warmup = max(1, int(updates * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(
            (s + 1) / warmup, max(0.05, (updates - s) / max(1, updates - warmup))
        ),
    )
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    start = time.perf_counter()
    history, best = [], float("inf")
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, tokens = 0.0, 0
        for i, batch in enumerate(loaders["train"]):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            n = (batch["labels"][:, 1:] != -100).sum().item()
            total += loss.item() * n
            tokens += n
            group_start = (i // cfg["gradient_accumulation"]) * cfg[
                "gradient_accumulation"
            ]
            group_size = min(
                cfg["gradient_accumulation"], len(loaders["train"]) - group_start
            )
            (loss / group_size).backward()
            if (i + 1) % cfg["gradient_accumulation"] == 0 or i + 1 == len(
                loaders["train"]
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        val_total, val_tokens = 0.0, 0
        with torch.inference_mode():
            for batch in loaders["validation"]:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                n = (batch["labels"][:, 1:] != -100).sum().item()
                val_total += model(**batch).loss.item() * n
                val_tokens += n
        record = dict(
            epoch=epoch + 1,
            train_loss=total / tokens,
            validation_loss=val_total / val_tokens,
            elapsed_seconds=time.perf_counter() - start,
        )
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["validation_loss"] < best:
            best = record["validation_loss"]
            model.save_pretrained(args.output / "adapter")
            tok.save_pretrained(args.output / "adapter")
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
    manifest = dict(
        config=cfg,
        selected_epoch=min(history, key=lambda r: r["validation_loss"])["epoch"],
        seconds=time.perf_counter() - start,
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        total_parameters=sum(p.numel() for p in model.parameters()),
        peak_vram_mb=torch.cuda.max_memory_allocated() / 1024**2
        if model.device.type == "cuda"
        else None,
        device=torch.cuda.get_device_name() if model.device.type == "cuda" else "CPU",
        python=platform.python_version(),
        torch=torch.__version__,
        data_sha256={
            s: hashlib.sha256((ROOT / f"data/{s}.jsonl").read_bytes()).hexdigest()
            for s in ["train", "validation", "test"]
        },
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    frozen = subprocess.check_output(
        [__import__("sys").executable, "-m", "pip", "freeze"], text=True
    )
    # The editable project path is machine-specific; reinstall it with pip install -e .
    frozen = (
        "\n".join(
            line
            for line in frozen.splitlines()
            if not line.startswith(("-e ", "# Editable"))
        )
        + "\n"
    )
    (args.output / "requirements-lock.txt").write_text(frozen)
    print(f"Saved best adapter and run evidence to {args.output}")


if __name__ == "__main__":
    main()
