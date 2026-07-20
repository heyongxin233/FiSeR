from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from baselines.models import DEFAULT_IMAGE_PROCESSOR, build_baseline_model
from fiser.inference import load_image_processor


def _device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(raw)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is visible")
    return result


def load_baseline_checkpoint(path: str | Path) -> tuple[dict, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Not a FiSeR baseline checkpoint: {path}")
    config = dict(payload.get("config") or {})
    config.setdefault("architecture", payload.get("architecture", "resnet50"))
    return config, payload["state_dict"]


def classify_one(
    image: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    processor_name: str | None = None,
    local_files_only: bool = False,
) -> dict:
    config, state_dict = load_baseline_checkpoint(checkpoint)
    device_obj = _device(device)
    processor_source = (
        processor_name
        or config.get("processor_name")
        or config.get("model_name")
        or DEFAULT_IMAGE_PROCESSOR
    )
    processor = load_image_processor(
        str(processor_source), processor_name=str(processor_source), local_files_only=local_files_only
    )
    model = build_baseline_model(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device_obj).eval()
    with Image.open(image) as source:
        inputs = processor(images=source.convert("RGB"), return_tensors="pt")
    pixels = inputs["pixel_values"].to(device_obj)
    with torch.inference_mode():
        logit_tensor = model(pixels).float().reshape(-1)[0]
        probability = torch.sigmoid(logit_tensor)
    logit = float(logit_tensor.item())
    probability_value = float(probability.item())
    return {
        "image": str(Path(image).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "architecture": str(config.get("architecture", "resnet50")),
        "device": str(device_obj),
        "score": logit,
        "probability_synthetic": probability_value,
        "prediction": "synthetic" if probability_value >= 0.5 else "natural",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one image with a local FiSeR baseline checkpoint")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = classify_one(
        image=args.image,
        checkpoint=args.checkpoint,
        device=args.device,
        processor_name=args.processor_name,
        local_files_only=args.local_files_only,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
