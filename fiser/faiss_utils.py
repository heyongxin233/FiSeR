from __future__ import annotations

from typing import Iterable

import numpy as np


def import_faiss_gpu():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "FiSeR evaluation requires faiss-gpu. Install the CUDA build with "
            "`python -m pip install faiss-gpu`."
        ) from exc
    if not hasattr(faiss, "get_num_gpus") or int(faiss.get_num_gpus()) < 1:
        raise RuntimeError(
            "The imported FAISS build has no visible CUDA device. "
            "Use faiss-gpu and set CUDA_VISIBLE_DEVICES to the intended GPUs."
        )
    if not hasattr(faiss, "GpuIndexFlatIP"):
        raise RuntimeError("The imported FAISS package is CPU-only; GpuIndexFlatIP is missing")
    return faiss


def build_sharded_inner_product_index(
    dimension: int,
    devices: Iterable[int] | None = None,
    use_float16: bool = True,
):
    """Build a FAISS GPU index over the currently visible CUDA devices."""
    faiss = import_faiss_gpu()
    visible = int(faiss.get_num_gpus())
    selected = list(range(visible)) if devices is None else [int(x) for x in devices]
    if not selected:
        raise ValueError("At least one FAISS GPU device is required")
    if any(index < 0 or index >= visible for index in selected):
        raise ValueError(f"FAISS device IDs must be in [0, {visible}); got {selected}")

    if len(selected) == 1:
        resources = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = selected[0]
        config.useFloat16 = bool(use_float16)
        return faiss.GpuIndexFlatIP(resources, int(dimension), config), [resources]

    shards = faiss.IndexShards(int(dimension), True, True)
    resources = []
    for device in selected:
        resource = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = device
        config.useFloat16 = bool(use_float16)
        shards.add_shard(faiss.GpuIndexFlatIP(resource, int(dimension), config))
        resources.append(resource)
    return shards, resources


def add_in_batches(index, vectors: np.ndarray, batch_size: int = 65536) -> None:
    vectors = np.asarray(vectors, dtype=np.float32)
    for start in range(0, len(vectors), max(1, int(batch_size))):
        index.add(vectors[start : start + max(1, int(batch_size))])


def visible_gpu_count() -> int:
    faiss = import_faiss_gpu()
    return int(faiss.get_num_gpus())
