from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fiser.inference import (
    DEFAULT_MODEL,
    classify_feature,
    encode_image,
    json_dump,
    load_fiser_model,
    load_head,
    load_image_processor,
    move_head,
)


def _parse_devices(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--faiss-devices must contain at least one integer")
    return values


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is visible")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify one image with FiSeR and a kNN, prototype, or linear head"
    )
    parser.add_argument("--image", required=True, help="Path to one RGB image")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", help="Legacy .pth file or converted Hugging Face directory")
    parser.add_argument("--processor-name", help="Optional processor repository/directory")
    parser.add_argument("--head", choices=["knn", "prototype", "linear"], default="knn")
    parser.add_argument("--head-checkpoint", help="Portable prototype/linear head .pt file")
    parser.add_argument("--database-pt", help="Embedding archive used as the kNN feature bank")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--protocol", choices=["legacy", "exponential"], default="legacy")
    parser.add_argument("--faiss-devices", default=None, help="Visible FAISS IDs, e.g. 0,1")
    parser.add_argument("--faiss-fp16", action="store_true")
    parser.add_argument("--device", default="cuda", help="cuda or cuda:N")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--strict-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict:
    device = _resolve_device(args.device)
    model = load_fiser_model(
        args.model_name,
        checkpoint=args.checkpoint,
        device=device,
        local_files_only=args.local_files_only,
        strict=args.strict_checkpoint,
    )
    processor = load_image_processor(
        args.model_name,
        checkpoint=args.checkpoint,
        processor_name=args.processor_name,
        local_files_only=args.local_files_only,
        image_size=args.image_size,
    )
    feature = encode_image(model, processor, args.image, layer=args.layer, device=device)

    head = None
    database = None
    if args.head in {"linear", "prototype"}:
        if not args.head_checkpoint:
            raise ValueError(f"--head-checkpoint is required for --head {args.head}")
        head = move_head(load_head(args.head_checkpoint, layer=args.layer), device)
        if head.get("layer") is not None and int(head["layer"]) != int(args.layer):
            raise ValueError(
                f"Head was built for layer {head['layer']}, but --layer {args.layer} was requested"
            )
    elif not args.database_pt:
        raise ValueError("--database-pt is required for --head knn")
    if args.database_pt:
        from fiser.inference import _load_archive

        database = _load_archive(args.database_pt)

    result = classify_feature(
        feature,
        head_type=args.head,
        head=head,
        database=database,
        layer=args.layer,
        k=args.k,
        temperature=args.temperature,
        protocol=args.protocol,
        devices=_parse_devices(args.faiss_devices),
        faiss_fp16=args.faiss_fp16,
        device=device,
    )
    result = {
        "image": str(Path(args.image).resolve()),
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "head": args.head,
        "layer": int(args.layer),
        "device": str(device),
        **result,
    }
    print(json_dump(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_dump(result) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main(parse_args())
