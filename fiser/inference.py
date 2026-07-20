"""Single-image FiSeR inference and lightweight head serialization."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from .faiss_utils import build_sharded_inner_product_index, import_faiss_gpu
from .heads import exponential_knn_probabilities, legacy_knn_probabilities, l2_normalize
from .models import FiSeREncoder, load_encoder


DEFAULT_MODEL = "heyongxin233/FiSeR-DINOv3-ViT-L16"


class SimpleImageProcessor:
    """ImageNet-style fallback for local DINO checkpoints without processor files."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = int(image_size)

    def __call__(
        self,
        images: Image.Image,
        return_tensors: str = "pt",
        do_resize: bool = True,
        do_center_crop: bool = True,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("SimpleImageProcessor only supports return_tensors='pt'")
        size = max(1, self.image_size)
        if do_resize or images.size != (size, size):
            width, height = images.size
            scale = size / min(width, height)
            resized = images.resize(
                (max(size, round(width * scale)), max(size, round(height * scale))),
                Image.Resampling.BICUBIC,
            )
        else:
            resized = images
        if do_center_crop or resized.size != (size, size):
            left = (resized.width - size) // 2
            top = (resized.height - size) // 2
            cropped = resized.crop((left, top, left + size, top + size))
        else:
            cropped = resized
        array = np.asarray(cropped, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return {"pixel_values": ((tensor - mean) / std).unsqueeze(0)}


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"Head field {name!r} must be a tensor")
    return value.detach().cpu().float()


def _unwrap_head_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Inference head must be a torch dictionary")
    for key in ("head", "state_dict", "model"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            merged = dict(payload)
            merged.update(nested)
            payload = merged
            break
    return dict(payload)


def load_head(path: str | Path, layer: int | None = None) -> dict[str, Any]:
    """Load a portable prototype/linear head checkpoint.

    ``scripts/build_inference_head.py`` writes this format, but plain
    ``nn.Linear`` state dictionaries with ``weight``/``bias`` are accepted as
    well so a trained binary head can be used without conversion.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload = _unwrap_head_payload(payload)
    # Historical FiSeR probe files store one linear state dict per hidden layer.
    probes = payload.get("probes")
    if isinstance(probes, dict):
        if layer is None:
            if len(probes) != 1:
                raise ValueError("A layer must be specified for a multi-layer probe checkpoint")
            selected_layer = next(iter(probes))
        else:
            selected_layer = layer
        nested = probes.get(selected_layer, probes.get(str(selected_layer)))
        if not isinstance(nested, dict):
            raise ValueError(f"Probe checkpoint has no head for layer {selected_layer}")
        merged = dict(payload)
        merged.update(nested)
        merged["layer"] = int(selected_layer)
        payload = merged
    raw_type = payload.get("head_type", payload.get("type"))

    if raw_type is None and "prototypes" in payload:
        raw_type = "prototype"
    if raw_type is None and any(str(key).endswith("weight") for key in payload):
        raw_type = "linear"
    head_type = str(raw_type or "").lower()
    if head_type not in {"linear", "prototype"}:
        raise ValueError(
            f"Unsupported inference head type {raw_type!r}; expected linear or prototype"
        )

    result: dict[str, Any] = {
        "head_type": head_type,
        "layer": int(payload["layer"]) if payload.get("layer") is not None else None,
        "classes": [str(value) for value in payload.get("classes", ["natural", "synthetic"])],
        "metadata": payload.get("metadata", {}),
    }
    if head_type == "prototype":
        prototypes = payload.get("prototypes")
        if prototypes is None:
            prototypes = payload.get("prototype")
        if prototypes is None:
            raise ValueError("Prototype head must contain a 'prototypes' tensor")
        prototypes = _as_tensor(prototypes, "prototypes")
        if prototypes.ndim != 2 or prototypes.shape[0] != 2:
            raise ValueError("Prototype head must have shape [2, dimension]")
        result["prototypes"] = torch.from_numpy(l2_normalize(prototypes.numpy()))
        return result

    weight = payload.get("weight")
    bias = payload.get("bias")
    if weight is None:
        for key, value in payload.items():
            if str(key).endswith("weight") and torch.is_tensor(value):
                weight = value
                break
    if bias is None:
        for key, value in payload.items():
            if str(key).endswith("bias") and torch.is_tensor(value):
                bias = value
                break
    if weight is None:
        raise ValueError("Linear head must contain a 'weight' tensor")
    weight = _as_tensor(weight, "weight")
    if weight.ndim != 2 or weight.shape[0] not in {1, 2}:
        raise ValueError("Linear head weight must have shape [1, dimension] or [2, dimension]")
    if bias is None:
        bias = torch.zeros(weight.shape[0], dtype=torch.float32)
    bias = _as_tensor(bias, "bias").reshape(-1)
    if bias.numel() != weight.shape[0]:
        raise ValueError("Linear head bias does not match the number of output classes")
    result["weight"] = weight
    result["bias"] = bias
    return result


def move_head(head: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move tensor fields of a loaded head to the inference device.

    Checkpoints are deliberately read on CPU, then moved explicitly after the
    CUDA device has been selected.  This keeps deserialization portable while
    ensuring the actual head computation stays beside the encoder.
    """
    target = torch.device(device)
    result = dict(head)
    for key in ("weight", "bias", "prototypes"):
        value = result.get(key)
        if torch.is_tensor(value):
            result[key] = value.to(device=target, dtype=torch.float32)
    return result


def load_image_processor(
    model_name: str,
    checkpoint: str | Path | None = None,
    processor_name: str | None = None,
    local_files_only: bool = False,
    image_size: int = 224,
):
    """Resolve a processor from a converted checkpoint, explicit name, or base model."""
    if processor_name:
        source = processor_name
    elif checkpoint and Path(checkpoint).is_dir():
        source = str(checkpoint)
    else:
        source = model_name
    try:
        return AutoImageProcessor.from_pretrained(source, local_files_only=local_files_only)
    except (OSError, ValueError) as exc:
        print(
            f"Warning: could not load an image processor from {source!r} ({exc}); "
            "using ImageNet normalization and center crop",
            file=sys.stderr,
        )
        return SimpleImageProcessor(image_size=image_size)


def load_fiser_model(
    model_name: str,
    checkpoint: str | Path | None = None,
    device: torch.device | str = "cpu",
    local_files_only: bool = False,
    strict: bool = True,
) -> FiSeREncoder:
    model = load_encoder(
        model_name,
        checkpoint=str(checkpoint) if checkpoint else None,
        local_files_only=local_files_only,
        strict=strict,
    )
    return model.to(device).eval()


def encode_image(
    model: FiSeREncoder,
    processor,
    image_path: str | Path,
    layer: int,
    device: torch.device | str,
    return_numpy: bool = False,
) -> torch.Tensor | np.ndarray:
    """Encode one image and keep the feature on ``device`` by default.

    ``return_numpy`` is provided for callers that need to serialize a feature;
    the command-line inference path leaves it as a CUDA tensor so head scoring
    does not make an unnecessary device round trip.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        hidden_states = model(pixel_values, output_hidden_states=True)
    if not isinstance(hidden_states, tuple):  # pragma: no cover - model contract guard
        raise RuntimeError("FiSeR encoder did not return hidden states")
    if layer < 0 or layer >= len(hidden_states):
        raise ValueError(f"Layer {layer} is unavailable; model returned {len(hidden_states)} layers")
    feature = torch.nn.functional.normalize(hidden_states[layer].float(), dim=-1)
    if return_numpy:
        return feature.detach().cpu().numpy().astype(np.float32, copy=False)
    return feature


def _feature_tensor(
    feature: np.ndarray | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a feature without pulling an existing CUDA tensor to CPU."""
    target = torch.device(device) if device is not None else None
    if torch.is_tensor(feature):
        tensor = feature.detach().to(dtype=torch.float32)
        return tensor.to(target) if target is not None else tensor
    tensor = torch.as_tensor(np.asarray(feature, dtype=np.float32), dtype=torch.float32)
    return tensor.to(target) if target is not None else tensor


def _linear_score(
    feature: np.ndarray | torch.Tensor,
    head: dict[str, Any],
    device: torch.device | str | None = None,
) -> float:
    feature_tensor = _feature_tensor(feature, device)
    if feature_tensor.ndim == 1:
        feature_tensor = feature_tensor.unsqueeze(0)
    if feature_tensor.ndim != 2:
        raise ValueError("Feature must have shape [1, dimension]")
    weight = head["weight"].to(device=feature_tensor.device, dtype=torch.float32)
    bias = head["bias"].to(device=feature_tensor.device, dtype=torch.float32)
    logits = feature_tensor @ weight.t() + bias
    if logits.shape[1] == 1:
        return float(logits[0, 0].detach().item())
    # A two-logit head follows the usual [natural, synthetic] class order.
    return float((logits[0, 1] - logits[0, 0]).detach().item())


def _prototype_score(
    feature: np.ndarray | torch.Tensor,
    head: dict[str, Any],
    device: torch.device | str | None = None,
) -> float:
    query = _feature_tensor(feature, device)
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if query.ndim != 2:
        raise ValueError("Feature must have shape [1, dimension]")
    query = torch.nn.functional.normalize(query, dim=-1)
    prototypes = torch.nn.functional.normalize(
        head["prototypes"].to(device=query.device, dtype=torch.float32), dim=-1
    )
    similarities = query @ prototypes.t()
    return float((similarities[0, 1] - similarities[0, 0]).detach().item())


def _load_archive(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("embeddings"), dict):
        raise ValueError(f"Not a FiSeR embedding archive: {path}")
    archive = dict(payload)
    archive["embeddings"] = {int(key): value for key, value in payload["embeddings"].items()}
    labels = torch.as_tensor(payload.get("labels", payload.get("binary_labels"))).long()
    if payload.get("binary_labels") is not None:
        binary_labels = torch.as_tensor(payload["binary_labels"]).long()
    else:
        classes = [str(value) for value in payload.get("classes", [])]
        if "nature" in classes:
            binary_labels = (labels != classes.index("nature")).long()
        elif set(torch.unique(labels).tolist()).issubset({0, 1}):
            binary_labels = labels
        else:
            raise ValueError(
                f"Embedding archive {path} has no binary_labels and no 'nature' class metadata"
            )
    archive["binary_labels"] = binary_labels
    archive["ids"] = torch.as_tensor(
        payload.get("ids", torch.arange(len(archive["binary_labels"])))
    ).long()
    return archive


def _knn_score(
    feature: np.ndarray | torch.Tensor,
    archive: dict[str, Any],
    layer: int,
    k: int,
    temperature: float,
    protocol: str,
    devices: list[int] | None,
    faiss_fp16: bool,
) -> tuple[float, list[dict[str, Any]]]:
    import_faiss_gpu()
    if layer not in archive["embeddings"]:
        raise ValueError(f"Layer {layer} is not present in the database archive")
    database = archive["embeddings"][layer]
    if torch.is_tensor(database):
        database_values = database.detach().float().cpu().numpy()
    else:
        database_values = np.asarray(database, dtype=np.float32)
    if database_values.ndim != 2 or len(database_values) == 0:
        raise ValueError("The kNN database must contain a non-empty [N, D] tensor")
    k = min(max(1, int(k)), len(database_values))
    index, resources = build_sharded_inner_product_index(
        database_values.shape[1], devices=devices, use_float16=faiss_fp16
    )
    try:
        index.add(database_values)
        if torch.is_tensor(feature):
            query = feature.detach().float().cpu().numpy()
        else:
            query = np.asarray(feature, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        similarities, indices = index.search(query, k)
        labels = archive["binary_labels"].detach().cpu().numpy().astype(np.int64)[
            np.maximum(indices, 0)
        ]
        if protocol == "legacy":
            probability = legacy_knn_probabilities(similarities, labels, temperature=temperature)[0, -1]
        elif protocol == "exponential":
            probability = exponential_knn_probabilities(similarities, labels, temperature=temperature)[0, -1]
        else:
            raise ValueError("protocol must be 'legacy' or 'exponential'")
        neighbor_ids = archive["ids"].detach().cpu().numpy().astype(np.int64)
        neighbors = [
            {
                "rank": int(rank + 1),
                "index": int(index_value),
                "id": int(neighbor_ids[max(0, int(index_value))]),
                "similarity": float(similarities[0, rank]),
                "label": "synthetic" if int(labels[0, rank]) else "natural",
            }
            for rank, index_value in enumerate(indices[0])
        ]
        return float(probability), neighbors
    finally:
        del resources
        del index


def classify_feature(
    feature: np.ndarray | torch.Tensor,
    head_type: str,
    head: dict[str, Any] | None = None,
    database: dict[str, Any] | None = None,
    layer: int = 18,
    k: int = 25,
    temperature: float = 0.05,
    protocol: str = "legacy",
    devices: list[int] | None = None,
    faiss_fp16: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Classify one normalized feature and return JSON-friendly values."""
    head_type = str(head_type).lower()
    score_device = torch.device(device) if device is not None else None
    if score_device is None and torch.is_tensor(feature):
        score_device = feature.device
    if head_type == "linear":
        if head is None:
            raise ValueError("A linear head checkpoint is required")
        raw_score = _linear_score(feature, head, device=score_device)
        probability = float(
            torch.sigmoid(torch.as_tensor(raw_score, dtype=torch.float32, device=score_device)).item()
        )
        details: dict[str, Any] = {}
    elif head_type == "prototype":
        if head is None:
            raise ValueError("A prototype head checkpoint is required")
        raw_score = _prototype_score(feature, head, device=score_device)
        probability = float(
            torch.sigmoid(torch.as_tensor(raw_score, dtype=torch.float32, device=score_device)).item()
        )
        details = {}
    elif head_type == "knn":
        if database is None:
            raise ValueError("A kNN database archive is required")
        probability, neighbors = _knn_score(
            feature,
            database,
            layer=layer,
            k=k,
            temperature=temperature,
            protocol=protocol,
            devices=devices,
            faiss_fp16=faiss_fp16,
        )
        raw_score = probability
        details = {"neighbors": neighbors}
    else:
        raise ValueError("head_type must be one of: knn, prototype, linear")
    return {
        "score": float(raw_score),
        "probability_synthetic": float(probability),
        "prediction": "synthetic" if probability >= 0.5 else "natural",
        **details,
    }


def json_dump(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
