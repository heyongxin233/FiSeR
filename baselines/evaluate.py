from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from baselines.models import DEFAULT_IMAGE_PROCESSOR, build_baseline_model
from fiser.data import FiSeRLMDBDataset
from fiser.distributed import all_gather_equal, barrier, cleanup_distributed, init_distributed
from fiser.inference import load_image_processor
from fiser.metrics import binary_metrics
from train import collate_samples, set_seed


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Not a FiSeR baseline checkpoint: {path}")
    config = dict(payload.get("config") or {})
    config.setdefault("architecture", payload.get("architecture", "resnet50"))
    return config, payload["state_dict"]


def evaluate(config: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> dict[str, Any] | None:
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
        config["dataset_lmdb"],
        processor=processor,
        nature_source=str(config.get("nature_source", "nature")),
        train=False,
        image_size=int(config.get("image_size", 224)),
        max_samples=config.get("max_samples"),
        seed=int(config.get("seed", 42)),
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 128)),
        sampler=sampler,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=ctx.device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 4)) > 0,
        collate_fn=collate_samples,
    )
    model = build_baseline_model(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(ctx.device).eval()

    all_scores: list[float] = []
    all_labels: list[int] = []
    all_ids: list[int] = []
    all_lmdb_indices: list[int] = []
    amp_enabled = ctx.device.type == "cuda" and str(config.get("precision", "bf16")) != "fp32"
    amp_dtype = torch.float16 if str(config.get("precision", "bf16")) == "fp16" else torch.bfloat16
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(ctx.device, non_blocking=True)
            with torch.autocast(ctx.device.type, dtype=amp_dtype, enabled=amp_enabled):
                scores = model(pixels).float()
            labels = batch["labels"].to(ctx.device, non_blocking=True)
            ids = batch["ids"].to(ctx.device, non_blocking=True)
            lmdb_indices = batch["lmdb_indices"].to(ctx.device, non_blocking=True)
            scores = all_gather_equal(scores)
            labels = all_gather_equal(labels)
            ids = all_gather_equal(ids)
            lmdb_indices = all_gather_equal(lmdb_indices)
            if ctx.is_main:
                all_scores.extend(scores.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_ids.extend(ids.cpu().tolist())
                all_lmdb_indices.extend(lmdb_indices.cpu().tolist())

    result: dict[str, Any] | None = None
    if ctx.is_main:
        keep: list[int] = []
        seen: set[int] = set()
        for index, lmdb_index in enumerate(all_lmdb_indices):
            if lmdb_index in seen:
                continue
            seen.add(lmdb_index)
            keep.append(index)
        scores = [all_scores[index] for index in keep]
        labels = [all_labels[index] for index in keep]
        result = {
            "architecture": str(config.get("architecture", "resnet50")),
            "dataset_lmdb": str(config["dataset_lmdb"]),
            "samples": len(keep),
            "metrics": binary_metrics(labels, scores, target_fpr=float(config.get("target_fpr", 0.05))),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if config.get("output"):
            output = Path(str(config["output"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    barrier()
    cleanup_distributed()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FiSeR baseline checkpoint on LMDB")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-lmdb", required=True)
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint_config, checkpoint_state = load_checkpoint(args.checkpoint)
    checkpoint_config.update(
        {
            "dataset_lmdb": args.dataset_lmdb,
            "output": args.output,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_samples": args.max_samples,
            "target_fpr": args.target_fpr,
        }
    )
    if args.local_files_only:
        checkpoint_config["local_files_only"] = True
    evaluate(checkpoint_config, checkpoint_state)
