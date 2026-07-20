from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fiser.faiss_utils import build_sharded_inner_product_index, import_faiss_gpu
from fiser.heads import (
    exponential_knn_probabilities,
    fit_parametric_head,
    legacy_knn_probabilities,
    parametric_scores,
    prototype_scores,
    select_best_k,
)
from fiser.metrics import binary_metrics


def load_archive(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        archive = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        archive = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(archive, dict) or "embeddings" not in archive:
        raise ValueError(f"Not a FiSeR embedding archive: {path}")
    embeddings = archive["embeddings"]
    if not isinstance(embeddings, dict) or not embeddings:
        raise ValueError(f"Embedding archive has no layer tensors: {path}")
    archive = dict(archive)
    archive["embeddings"] = {int(layer): tensor for layer, tensor in embeddings.items()}
    archive["ids"] = torch.as_tensor(archive["ids"]).long()
    archive["labels"] = torch.as_tensor(archive["labels"]).long()
    classes = [str(value) for value in archive.get("classes", [])]
    archive["classes"] = classes
    if "binary_labels" in archive:
        archive["binary_labels"] = torch.as_tensor(archive["binary_labels"]).long()
    elif classes:
        nature_index = classes.index("nature") if "nature" in classes else None
        if nature_index is None:
            raise ValueError(f"Embedding archive {path} has no 'nature' class")
        archive["binary_labels"] = (archive["labels"] != nature_index).long()
    else:
        raise ValueError(f"Embedding archive {path} has no binary_labels/classes metadata")
    return archive


def _layer_tensor(archive: dict[str, Any], layer: int) -> np.ndarray:
    tensor = archive["embeddings"][layer]
    if tensor.ndim != 2:
        raise ValueError(f"Layer {layer} must have shape [N, D], got {tuple(tensor.shape)}")
    return tensor


def _search_gpu(
    database: np.ndarray | torch.Tensor,
    queries: np.ndarray | torch.Tensor,
    max_k: int,
    devices: list[int] | None,
    add_batch_size: int,
    query_batch_size: int,
    faiss_fp16: bool,
):
    if isinstance(database, torch.Tensor):
        dimension = int(database.shape[1])
    else:
        dimension = int(database.shape[1])
    index, resources = build_sharded_inner_product_index(
        dimension, devices=devices, use_float16=faiss_fp16
    )
    try:
        # FAISS IndexShards with successive_ids=True requires one logical add
        # pass. Keep query search chunked, but submit this layer in one call so
        # shard-local IDs remain globally aligned.
        if isinstance(database, torch.Tensor):
            database_values = database.float().numpy()
        else:
            database_values = np.asarray(database, dtype=np.float32)
        index.add(database_values)
        all_scores: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        for start in range(0, len(queries), max(1, query_batch_size)):
            values = queries[start : start + max(1, query_batch_size)]
            if isinstance(values, torch.Tensor):
                values = values.float().numpy()
            scores, indices = index.search(np.asarray(values, dtype=np.float32), int(max_k))
            all_scores.append(np.asarray(scores))
            all_indices.append(np.asarray(indices))
        return np.concatenate(all_scores), np.concatenate(all_indices)
    finally:
        # Keep resources alive through search and release the index afterwards.
        del resources
        del index


def _neighbor_labels(indices: np.ndarray, database_binary: np.ndarray) -> np.ndarray:
    safe_indices = np.maximum(indices, 0)
    return database_binary[safe_indices]


def evaluate_knn(
    database: dict[str, Any],
    query: dict[str, Any],
    max_k: int,
    temperature: float,
    devices: list[int] | None,
    add_batch_size: int,
    query_batch_size: int,
    protocol: str,
    target_fpr: float,
    requested_layers: list[int] | None,
    faiss_fp16: bool,
) -> dict[str, Any]:
    database_layers = set(database["embeddings"])
    query_layers = set(query["embeddings"])
    layers = sorted(database_layers & query_layers)
    if requested_layers is not None:
        requested = {int(layer) for layer in requested_layers}
        layers = [layer for layer in layers if layer in requested]
    if not layers:
        raise ValueError(f"No common layers: database={sorted(database_layers)}, query={sorted(query_layers)}")
    database_binary = database["binary_labels"].numpy().astype(np.int64)
    query_binary = query["binary_labels"].numpy().astype(np.int64)
    if len(database_binary) != len(database["ids"]):
        raise ValueError("Database labels and embeddings have different lengths")
    if len(query_binary) != len(query["ids"]):
        raise ValueError("Query labels and embeddings have different lengths")

    per_layer: list[dict[str, Any]] = []
    for layer in layers:
        scores, indices = _search_gpu(
            _layer_tensor(database, layer),
            _layer_tensor(query, layer),
            max_k=max_k,
            devices=devices,
            add_batch_size=add_batch_size,
            query_batch_size=query_batch_size,
            faiss_fp16=faiss_fp16,
        )
        neighbors = _neighbor_labels(indices, database_binary)
        if protocol == "legacy":
            probabilities = legacy_knn_probabilities(scores, neighbors, temperature=temperature)
        elif protocol == "exponential":
            probabilities = exponential_knn_probabilities(scores, neighbors, temperature=temperature)
        else:
            raise ValueError("protocol must be 'legacy' or 'exponential'")
        best = select_best_k(query_binary, probabilities, target_fpr=target_fpr)
        row = {"layer": int(layer), "k": int(best.k), **best.metrics}
        per_layer.append(row)
        print(
            f"layer={layer} best_k={best.k} auroc={best.metrics['auroc']:.6f} "
            f"tpr@fpr={best.metrics['tpr_at_fpr']:.6f}",
            flush=True,
        )
    best = max(per_layer, key=lambda row: row["auroc"])
    return {
        "head": "knn",
        "protocol": protocol,
        "best": best,
        "last": per_layer[-1],
        "layers": per_layer,
        "database": database.get("metadata", {}),
        "query": query.get("metadata", {}),
    }


def sample_support(
    archive: dict[str, Any], shots_per_source: int | None, shots_per_class: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(int(seed))
    source_labels = archive["labels"].numpy().astype(np.int64)
    binary_labels = archive["binary_labels"].numpy().astype(np.int64)
    groups = source_labels if shots_per_source is not None else binary_labels
    shots = shots_per_source if shots_per_source is not None else shots_per_class
    if shots is None:
        raise ValueError("Set --shots-per-source or --shots-per-class for few-shot evaluation")
    selected: list[int] = []
    for group in sorted(np.unique(groups).tolist()):
        candidates = np.flatnonzero(groups == group).tolist()
        rng.shuffle(candidates)
        selected.extend(candidates[: min(int(shots), len(candidates))])
    selected = sorted(selected)
    return np.asarray(selected, dtype=np.int64), binary_labels[selected]


def evaluate_fewshot(
    archive: dict[str, Any],
    head: str,
    shots_per_source: int | None,
    shots_per_class: int | None,
    seed: int,
    target_fpr: float,
    requested_layers: list[int] | None = None,
    devices: list[int] | None = None,
    max_k: int = 51,
    temperature: float = 0.05,
    protocol: str = "legacy",
    query_batch_size: int = 8192,
    faiss_fp16: bool = False,
    svm_backend: str = "sklearn",
) -> dict[str, Any]:
    support_indices, support_labels = sample_support(
        archive, shots_per_source=shots_per_source, shots_per_class=shots_per_class, seed=seed
    )
    all_indices = np.arange(len(archive["ids"]))
    query_indices = np.setdiff1d(all_indices, support_indices, assume_unique=False)
    query_labels = archive["binary_labels"].numpy()[query_indices]
    layers: list[dict[str, Any]] = []
    layers_to_run = sorted(archive["embeddings"])
    if requested_layers is not None:
        requested = {int(layer) for layer in requested_layers}
        layers_to_run = [layer for layer in layers_to_run if layer in requested]
    if not layers_to_run:
        raise ValueError("No requested few-shot layers are present in the archive")
    for layer in layers_to_run:
        values = _layer_tensor(archive, layer)
        support = values[support_indices].float().numpy()
        queries = values[query_indices].float().numpy()
        row: dict[str, Any]
        if head == "knn":
            similarities, indices = _search_gpu(
                support,
                queries,
                max_k=min(int(max_k), len(support)),
                devices=devices,
                add_batch_size=len(support),
                query_batch_size=query_batch_size,
                faiss_fp16=faiss_fp16,
            )
            neighbor_labels = support_labels[indices]
            if protocol == "legacy":
                probabilities = legacy_knn_probabilities(
                    similarities, neighbor_labels, temperature=temperature
                )
            elif protocol == "exponential":
                probabilities = exponential_knn_probabilities(
                    similarities, neighbor_labels, temperature=temperature
                )
            else:
                raise ValueError("protocol must be 'legacy' or 'exponential'")
            selection = select_best_k(query_labels, probabilities, target_fpr=target_fpr)
            row = {"layer": int(layer), "k": int(selection.k), **selection.metrics}
        elif head == "prototype":
            scores = prototype_scores(support, support_labels, queries)
            row = {"layer": int(layer), **binary_metrics(query_labels, scores, target_fpr=target_fpr)}
        elif head in {"linear", "svm"}:
            model = fit_parametric_head(
                head,
                support,
                support_labels,
                seed=seed,
                svm_backend=svm_backend,
            )
            scores = parametric_scores(model, queries)
            row = {"layer": int(layer), **binary_metrics(query_labels, scores, target_fpr=target_fpr)}
        else:
            raise ValueError("few-shot head must be knn, prototype, linear, or svm")
        layers.append(row)
        print(f"layer={layer} auroc={row['auroc']:.6f} tpr@fpr={row['tpr_at_fpr']:.6f}")
    best = max(layers, key=lambda row: row["auroc"])
    return {
        "head": head,
        "shots_per_source": shots_per_source,
        "shots_per_class": shots_per_class,
        "support_size": int(len(support_indices)),
        "query_size": int(len(query_indices)),
        "best": best,
        "layers": layers,
    }


def evaluate_fewshot_trials(
    archive: dict[str, Any],
    num_trials: int,
    **kwargs: Any,
) -> dict[str, Any]:
    if int(num_trials) < 1:
        raise ValueError("num_trials must be positive")
    base_seed = int(kwargs.pop("seed"))
    trials = [
        evaluate_fewshot(archive, seed=base_seed + trial, **kwargs)
        for trial in range(int(num_trials))
    ]
    metric_names = ("auroc", "average_precision", "tpr_at_fpr")
    summary: dict[str, Any] = {}
    for metric in metric_names:
        values = np.asarray([trial["best"][metric] for trial in trials], dtype=np.float64)
        summary[metric] = {"mean": float(values.mean()), "std": float(values.std())}
    summary["selected_layers"] = [int(trial["best"]["layer"]) for trial in trials]
    if all("k" in trial["best"] for trial in trials):
        summary["selected_k"] = [int(trial["best"]["k"]) for trial in trials]
    return {
        "head": str(kwargs["head"]),
        "num_trials": int(num_trials),
        "base_seed": base_seed,
        "shots_per_source": kwargs.get("shots_per_source"),
        "shots_per_class": kwargs.get("shots_per_class"),
        "summary": summary,
        "trials": trials,
    }


def parse_devices(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FiSeR embeddings with GPU FAISS and lightweight heads")
    parser.add_argument("--database-pt")
    parser.add_argument("--test-pt")
    parser.add_argument("--fewshot-pt", help="One archive from which support/query samples are split")
    parser.add_argument("--head", choices=["knn", "prototype", "linear", "svm"], default="knn")
    parser.add_argument("--shots-per-source", type=int)
    parser.add_argument("--shots-per-class", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--svm-backend", choices=["sklearn", "cuml", "auto"], default="sklearn")
    parser.add_argument("--max-k", type=int, default=51)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--protocol", choices=["legacy", "exponential"], default="legacy")
    parser.add_argument("--faiss-devices", default=None, help="Visible FAISS device IDs, e.g. 0,1")
    parser.add_argument("--add-batch-size", type=int, default=65536)
    parser.add_argument("--query-batch-size", type=int, default=8192)
    parser.add_argument("--faiss-fp16", action="store_true", help="Use half-precision GPU index storage")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict[str, Any]:
    # Explicitly fail before loading multi-gigabyte archives if FAISS is CPU-only.
    import_faiss_gpu()
    devices = parse_devices(args.faiss_devices)
    if args.fewshot_pt:
        archive = load_archive(args.fewshot_pt)
        fewshot_kwargs = dict(
            head=args.head,
            shots_per_source=args.shots_per_source,
            shots_per_class=args.shots_per_class,
            seed=args.seed,
            target_fpr=args.target_fpr,
            requested_layers=args.layers,
            devices=devices,
            max_k=args.max_k,
            temperature=args.temperature,
            protocol=args.protocol,
            query_batch_size=args.query_batch_size,
            faiss_fp16=args.faiss_fp16,
            svm_backend=args.svm_backend,
        )
        if args.num_trials == 1:
            result = evaluate_fewshot(archive, **fewshot_kwargs)
        else:
            result = evaluate_fewshot_trials(
                archive,
                num_trials=args.num_trials,
                **fewshot_kwargs,
            )
    else:
        if not args.database_pt or not args.test_pt:
            raise ValueError("Set --database-pt and --test-pt, or use --fewshot-pt")
        database = load_archive(args.database_pt)
        query = load_archive(args.test_pt)
        if args.head != "knn":
            raise ValueError("Cross-domain archives currently support --head knn; use --fewshot-pt for refitting")
        result = evaluate_knn(
            database,
            query,
            max_k=args.max_k,
            temperature=args.temperature,
            devices=devices,
            add_batch_size=args.add_batch_size,
            query_batch_size=args.query_batch_size,
            protocol=args.protocol,
            target_fpr=args.target_fpr,
            requested_layers=args.layers,
            faiss_fp16=args.faiss_fp16,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main(parse_args())
