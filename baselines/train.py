from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from baselines.models import DEFAULT_IMAGE_PROCESSOR, build_baseline_model
from fiser.config import apply_overrides, load_yaml, require_keys
from fiser.data import FiSeRLMDBDataset
from fiser.distributed import barrier, cleanup_distributed, init_distributed
from fiser.inference import load_image_processor
from train import collate_samples, cosine_lr, set_seed


def train(config: dict[str, Any]) -> None:
    require_keys(config, "train_lmdb", "output_dir")
    ctx = init_distributed()
    set_seed(int(config.get("seed", 42)), ctx.rank)
    processor_name = str(
        config.get(
            "processor_name",
            config.get("model_name", DEFAULT_IMAGE_PROCESSOR),
        )
    )
    processor = load_image_processor(
        processor_name,
        processor_name=processor_name,
        local_files_only=bool(config.get("local_files_only", False)),
        image_size=int(config.get("image_size", 224)),
    )
    dataset = FiSeRLMDBDataset(
        config["train_lmdb"],
        processor=processor,
        source_map=None,
        train=True,
        image_size=int(config.get("image_size", 224)),
        max_samples=config.get("max_samples"),
        seed=int(config.get("seed", 42)),
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
        seed=int(config.get("seed", 42)),
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("per_device_batch_size", 64)),
        sampler=sampler,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=ctx.device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 4)) > 0,
        drop_last=True,
        collate_fn=collate_samples,
    )
    model: torch.nn.Module = build_baseline_model(config).to(ctx.device)
    if ctx.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = int(config.get("epochs", 5))
    accumulation = max(1, int(config.get("gradient_accumulation_steps", 1)))
    total_steps = max(1, math.ceil(len(loader) / accumulation) * epochs)
    amp_enabled = ctx.device.type == "cuda" and str(config.get("precision", "bf16")) != "fp32"
    amp_dtype = torch.float16 if str(config.get("precision", "bf16")) == "fp16" else torch.bfloat16
    scaler_enabled = amp_dtype == torch.float16 and amp_enabled
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True) if ctx.is_main else None
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        iterator = tqdm(loader, disable=not ctx.is_main, desc=f"baseline epoch {epoch + 1}/{epochs}")
        for batch_index, batch in enumerate(iterator):
            pixels = batch["pixel_values"].to(ctx.device, non_blocking=True)
            labels = batch["labels"].float().to(ctx.device, non_blocking=True)
            with torch.autocast(ctx.device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(pixels)
                loss = criterion(logits, labels) / accumulation
            if scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.detach().item()) * accumulation
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                if scaler_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                lr = cosine_lr(
                    global_step,
                    int(config.get("warmup_steps", 1000)),
                    total_steps,
                    float(config.get("learning_rate", 1e-3)),
                    float(config.get("min_learning_rate", 1e-6)),
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if ctx.is_main and global_step % int(config.get("log_every_steps", 10)) == 0:
                iterator.set_postfix(loss=f"{running / (batch_index + 1):.4f}")
            if config.get("max_steps") is not None and global_step >= int(config["max_steps"]):
                break
        if ctx.is_main:
            print(f"epoch={epoch + 1} loss={running / max(1, batch_index + 1):.6f}", flush=True)
        if config.get("max_steps") is not None and global_step >= int(config["max_steps"]):
            break

    if ctx.is_main:
        model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
        torch.save(
            {
                "state_dict": model_to_save.state_dict(),
                "architecture": str(config.get("architecture", "resnet50")),
                "binary_label_definition": "0=natural, 1=synthetic",
                "config": config,
            },
            output_dir / f"{str(config.get('architecture', 'resnet50')).lower()}_final.pt",
        )
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    barrier()
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a FiSeR paper baseline")
    parser.add_argument("--config", default="baselines/configs/resnet50_wildfake.yaml")
    parser.add_argument("--train-lmdb")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = apply_overrides(load_yaml(args.config), args.override)
    if args.train_lmdb is not None:
        config["train_lmdb"] = args.train_lmdb
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train(config)
