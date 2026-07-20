from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from fiser.config import apply_overrides, load_yaml, require_keys
from fiser.data import FiSeRLMDBDataset, load_source_map
from fiser.distributed import all_gather_equal, barrier, cleanup_distributed, init_distributed
from fiser.inference import load_image_processor
from fiser.losses import HierarchicalContrastiveLoss
from fiser.models import FiSeREncoder, load_encoder


def set_seed(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "sources": torch.tensor([sample["source_id"] for sample in samples], dtype=torch.long),
        "ids": torch.tensor([sample["sample_id"] for sample in samples], dtype=torch.long),
        "lmdb_indices": torch.tensor([sample["lmdb_index"] for sample in samples], dtype=torch.long),
    }


def cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return min_lr
    progress = min(1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def resolve_accumulation_steps(config: dict[str, Any], world_size: int) -> int:
    target = config.get("global_batch_size")
    if target is None:
        return max(1, int(config.get("gradient_accumulation_steps", 1)))
    per_device = int(config.get("per_device_batch_size", 8))
    micro_global = per_device * max(1, int(world_size))
    target = int(target)
    if target < micro_global or target % micro_global != 0:
        raise ValueError(
            f"global_batch_size={target} must be a positive multiple of "
            f"per_device_batch_size * world_size={micro_global}"
        )
    return target // micro_global


def save_model(model: torch.nn.Module, output_dir: Path, config: dict[str, Any], ctx) -> None:
    if not ctx.is_main:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    assert isinstance(unwrapped, FiSeREncoder)
    unwrapped.save_pretrained(output_dir)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def train(config: dict[str, Any]) -> None:
    require_keys(config, "model_name", "train_lmdb", "output_dir")
    ctx = init_distributed()
    set_seed(int(config.get("seed", 42)), ctx.rank)
    if ctx.is_main:
        print(f"FiSeR training on {ctx.world_size} process(es), device={ctx.device}", flush=True)

    model_name = str(config["model_name"])
    local_files_only = bool(config.get("local_files_only", False))
    processor = load_image_processor(
        model_name,
        processor_name=config.get("processor_name"),
        local_files_only=local_files_only,
        image_size=int(config.get("image_size", 224)),
    )
    source_map = load_source_map(config.get("source_map"))
    dataset = FiSeRLMDBDataset(
        config["train_lmdb"],
        processor=processor,
        source_map=source_map,
        nature_source=str(config.get("nature_source", "nature")),
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
        batch_size=int(config.get("per_device_batch_size", 8)),
        sampler=sampler,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=ctx.device.type == "cuda",
        drop_last=True,
        persistent_workers=int(config.get("num_workers", 4)) > 0,
        collate_fn=collate_samples,
    )
    if len(loader) == 0:
        raise ValueError("The training loader is empty; reduce batch size or check the LMDB")

    init_checkpoint = config.get("init_checkpoint")
    model = load_encoder(
        model_name,
        checkpoint=init_checkpoint,
        local_files_only=local_files_only,
        strict=bool(config.get("strict_checkpoint", True)),
    ).to(ctx.device)
    if bool(config.get("gradient_checkpointing", False)):
        model.enable_gradient_checkpointing()
    if ctx.world_size > 1:
        static_graph = bool(config.get("ddp_static_graph", True))
        model = DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            # DINOv3's mask token is consistently unused by the image forward,
            # while checkpointed transformer blocks need static-graph DDP.
            find_unused_parameters=False,
            static_graph=static_graph,
        )

    objective = HierarchicalContrastiveLoss(
        temperature=float(config.get("temperature", 0.07)),
        coarse_weight=float(config.get("coarse_weight", 1.0)),
        fine_weight=float(config.get("fine_weight", 1.0)),
    ).to(ctx.device)
    trainable = list(model.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate", 3e-5)),
        betas=(float(config.get("adam_beta1", 0.9)), float(config.get("adam_beta2", 0.99))),
        eps=float(config.get("adam_epsilon", 1e-6)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    accumulation = resolve_accumulation_steps(config, ctx.world_size)
    if ctx.is_main:
        micro_global = int(config.get("per_device_batch_size", 8)) * ctx.world_size
        print(
            f"contrastive_batch={micro_global} optimizer_batch={micro_global * accumulation} "
            f"gradient_accumulation={accumulation}",
            flush=True,
        )
    epochs = int(config.get("epochs", 20))
    max_steps = config.get("max_steps")
    estimated_steps = max(1, math.ceil(len(loader) / accumulation) * epochs)
    if max_steps is not None:
        estimated_steps = min(estimated_steps, int(max_steps))
    scaler_enabled = str(config.get("precision", "bf16")).lower() == "fp16" and ctx.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    amp_dtype = torch.float16 if scaler_enabled else torch.bfloat16
    use_amp = ctx.device.type == "cuda" and str(config.get("precision", "bf16")).lower() != "fp32"

    output_dir = Path(str(config["output_dir"]))
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        running = {"loss": 0.0, "coarse_loss": 0.0, "fine_loss": 0.0}
        iterator = tqdm(loader, disable=not ctx.is_main, desc=f"epoch {epoch + 1}/{epochs}")
        for batch_index, batch in enumerate(iterator):
            pixel_values = batch["pixel_values"].to(ctx.device, non_blocking=True)
            labels = batch["labels"].to(ctx.device, non_blocking=True)
            sources = batch["sources"].to(ctx.device, non_blocking=True)
            local_batch = labels.shape[0]
            with torch.autocast(
                device_type=ctx.device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                anchors = model(pixel_values)
                bank = all_gather_equal(anchors)
                bank_labels = all_gather_equal(labels)
                bank_sources = all_gather_equal(sources)
                self_indices = torch.arange(local_batch, device=ctx.device) + ctx.rank * local_batch
                values = objective(
                    anchors,
                    bank,
                    labels,
                    bank_labels,
                    sources,
                    bank_sources,
                    self_bank_indices=self_indices,
                )
                loss = values["loss"] / accumulation

            if scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            for key in running:
                running[key] += float(values[key].detach().float().item())

            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                if scaler_enabled:
                    scaler.unscale_(optimizer)
                max_norm = float(config.get("max_grad_norm", 1.0))
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                lr = cosine_lr(
                    global_step,
                    int(config.get("warmup_steps", 2000)),
                    estimated_steps,
                    float(config.get("learning_rate", 3e-5)),
                    float(config.get("min_learning_rate", 5e-6)),
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
                    denominator = batch_index + 1
                    iterator.set_postfix(
                        loss=f"{running['loss'] / denominator:.4f}",
                        coarse=f"{running['coarse_loss'] / denominator:.4f}",
                        fine=f"{running['fine_loss'] / denominator:.4f}",
                        lr=f"{lr:.2e}",
                    )
                if max_steps is not None and global_step >= int(max_steps):
                    break
        if ctx.is_main:
            denominator = max(1, batch_index + 1)
            print(
                f"epoch={epoch + 1} loss={running['loss'] / denominator:.6f} "
                f"coarse={running['coarse_loss'] / denominator:.6f} "
                f"fine={running['fine_loss'] / denominator:.6f}",
                flush=True,
            )
        if (epoch + 1) % int(config.get("save_every_epochs", 1)) == 0:
            barrier()
            save_model(model, output_dir / f"epoch_{epoch + 1}", config, ctx)
            barrier()
        if max_steps is not None and global_step >= int(max_steps):
            break

    barrier()
    save_model(model, output_dir / "final", config, ctx)
    barrier()
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FiSeR with hierarchical contrastive learning")
    parser.add_argument("--config", default="configs/fiser_wildfake.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value; can be passed more than once",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    settings = apply_overrides(load_yaml(arguments.config), arguments.override)
    try:
        train(settings)
    finally:
        cleanup_distributed()
