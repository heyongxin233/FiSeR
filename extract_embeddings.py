from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from fiser.config import load_yaml, require_keys
from fiser.data import FiSeRLMDBDataset, derive_source_map, load_source_map
from fiser.distributed import all_gather_equal, barrier, cleanup_distributed, init_distributed
from fiser.inference import load_image_processor
from fiser.models import load_encoder
from train import collate_samples, set_seed


def save_embedding_archive(
    path: Path,
    embeddings: dict[int, torch.Tensor],
    ids: list[int],
    binary_labels: list[int],
    source_labels: list[int],
    classes: list[str],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = {
        "embeddings": embeddings,
        "ids": torch.tensor(ids, dtype=torch.long),
        # `labels` keeps compatibility with the original Image-Detection files:
        # it stores the source/model ID, while binary_labels is explicit.
        "labels": torch.tensor(source_labels, dtype=torch.long),
        "binary_labels": torch.tensor(binary_labels, dtype=torch.long),
        "classes": classes,
        "metadata": metadata,
    }
    torch.save(archive, path)


def extract(config: dict[str, Any]) -> None:
    require_keys(config, "model_name", "dataset_lmdb", "output")
    ctx = init_distributed()
    set_seed(int(config.get("seed", 42)), ctx.rank)
    model_name = str(config["model_name"])
    local_files_only = bool(config.get("local_files_only", False))
    processor = load_image_processor(
        model_name,
        processor_name=config.get("processor_name"),
        local_files_only=local_files_only,
        image_size=int(config.get("image_size", 224)),
    )
    source_map = load_source_map(config.get("source_map"))
    if source_map is None and bool(config.get("derive_source_map", False)):
        if ctx.is_main:
            source_map = derive_source_map(
                config["dataset_lmdb"], nature_source=str(config.get("nature_source", "nature"))
            )
        if dist.is_available() and dist.is_initialized():
            payload = [source_map]
            dist.broadcast_object_list(payload, src=0)
            source_map = payload[0]
    dataset = FiSeRLMDBDataset(
        config["dataset_lmdb"],
        processor=processor,
        source_map=source_map,
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
        batch_size=int(config.get("batch_size", 64)),
        sampler=sampler,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=ctx.device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 4)) > 0,
        collate_fn=collate_samples,
    )
    model = load_encoder(
        model_name,
        checkpoint=config.get("checkpoint"),
        local_files_only=local_files_only,
        strict=bool(config.get("strict_checkpoint", True)),
    ).to(ctx.device)
    model.eval()

    requested_layers = [int(x) for x in config.get("layers", [15, 16, 17, 18, 19, 20, 21, 22, 23])]
    local_embeddings: dict[int, list[torch.Tensor]] = {layer: [] for layer in requested_layers}
    local_ids: list[int] = []
    local_binary: list[int] = []
    local_sources: list[int] = []
    local_lmdb_indices: list[int] = []
    with torch.inference_mode():
        iterator = tqdm(loader, disable=not ctx.is_main, desc="extract")
        for batch in iterator:
            pixel_values = batch["pixel_values"].to(ctx.device, non_blocking=True)
            labels = batch["labels"].to(ctx.device, non_blocking=True)
            sources = batch["sources"].to(ctx.device, non_blocking=True)
            ids = batch["ids"].to(ctx.device, non_blocking=True)
            lmdb_indices = batch["lmdb_indices"].to(ctx.device, non_blocking=True)
            hidden_states = model(pixel_values, output_hidden_states=True)
            assert isinstance(hidden_states, tuple)
            if any(layer < 0 or layer >= len(hidden_states) for layer in requested_layers):
                raise ValueError(
                    f"Requested layers {requested_layers}, but the model has {len(hidden_states)} hidden states"
                )
            selected = torch.stack([hidden_states[layer] for layer in requested_layers], dim=1)
            selected = torch.nn.functional.normalize(selected.float(), dim=-1)

            selected = all_gather_equal(selected)
            labels = all_gather_equal(labels)
            sources = all_gather_equal(sources)
            ids = all_gather_equal(ids)
            lmdb_indices = all_gather_equal(lmdb_indices)
            if ctx.is_main:
                for index, layer in enumerate(requested_layers):
                    local_embeddings[layer].append(selected[:, index].cpu().to(torch.bfloat16))
                local_binary.extend(labels.cpu().tolist())
                local_sources.extend(sources.cpu().tolist())
                local_ids.extend(ids.cpu().tolist())
                local_lmdb_indices.extend(lmdb_indices.cpu().tolist())

    if ctx.is_main:
        stacked = {layer: torch.cat(chunks, dim=0) for layer, chunks in local_embeddings.items()}
        # DistributedSampler pads the final batch. Preserve one row per sample ID.
        keep: list[int] = []
        seen: set[int] = set()
        for index, lmdb_index in enumerate(local_lmdb_indices):
            if int(lmdb_index) in seen:
                continue
            seen.add(int(lmdb_index))
            keep.append(index)
        stacked = {layer: values[keep] for layer, values in stacked.items()}
        ids = [int(local_ids[index]) for index in keep]
        binary = [int(local_binary[index]) for index in keep]
        sources = [int(local_sources[index]) for index in keep]
        if source_map:
            classes = [name for name, _ in sorted(source_map.items(), key=lambda item: item[1])]
        else:
            classes = []
        metadata = {
            "model_name": model_name,
            "checkpoint": config.get("checkpoint"),
            "dataset_lmdb": str(config["dataset_lmdb"]),
            "layers": requested_layers,
            "normalized": True,
            "binary_label_definition": "0=natural, 1=synthetic",
        }
        save_embedding_archive(
            Path(str(config["output"])), stacked, ids, binary, sources, classes, metadata
        )
        print(f"Saved {len(ids)} samples and layers {requested_layers} to {config['output']}")
    barrier()
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract normalized FiSeR layer embeddings")
    parser.add_argument("--config")
    parser.add_argument("--model-name")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset-lmdb")
    parser.add_argument("--source-map")
    parser.add_argument("--derive-source-map", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings: dict[str, Any] = {}
    if args.config:
        settings.update(load_yaml(args.config))
    for key, value in {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "dataset_lmdb": args.dataset_lmdb,
        "source_map": args.source_map,
        "derive_source_map": args.derive_source_map,
        "output": args.output,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_samples": args.max_samples,
    }.items():
        if value is not None:
            settings[key] = value
    if args.local_files_only:
        settings["local_files_only"] = True
    settings.setdefault("seed", args.seed)
    extract(settings)
