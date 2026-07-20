"""Convert a legacy FiSeR `.pth` state dict into a Hugging Face directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoImageProcessor

from fiser.models import FiSeREncoder, _unwrap_state_dict


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload: Any = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return _unwrap_state_dict(payload)


def _checkpoint_dtype(state_dict: dict[str, torch.Tensor]) -> torch.dtype | None:
    dtypes = {value.dtype for value in state_dict.values() if value.is_floating_point()}
    if len(dtypes) > 1:
        raise ValueError(f"Checkpoint has multiple floating-point dtypes: {sorted(map(str, dtypes))}")
    return next(iter(dtypes), None)


def convert(
    checkpoint: str | Path,
    base_model: str,
    output: str | Path,
    processor_model: str | None = None,
    local_files_only: bool = False,
) -> Path:
    checkpoint_path = Path(checkpoint)
    output_path = Path(output)
    state_dict = _load_state_dict(checkpoint_path)
    if not state_dict:
        raise ValueError(f"No tensor parameters found in checkpoint: {checkpoint_path}")

    model = FiSeREncoder(base_model, local_files_only=local_files_only)
    dtype = _checkpoint_dtype(state_dict)
    if dtype is not None:
        # Legacy FiSeR checkpoints are BF16. Cast before loading so the exported
        # safetensors preserve the source dtype instead of silently becoming FP32.
        model.encoder.to(dtype=dtype)
    incompatible = model.encoder.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"Checkpoint does not match the base model: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    processor_source = processor_model or base_model
    processor = AutoImageProcessor.from_pretrained(
        processor_source,
        local_files_only=local_files_only,
    )
    processor.save_pretrained(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Legacy FiSeR .pth checkpoint")
    parser.add_argument("--base-model", required=True, help="DINOv3 model or compatible HF directory")
    parser.add_argument("--processor-model", help="Model/repository providing the image processor")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = convert(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        output=args.output,
        processor_model=args.processor_model,
        local_files_only=args.local_files_only,
    )
    print(f"Saved Hugging Face checkpoint to {result}")
