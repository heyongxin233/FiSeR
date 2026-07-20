from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from fiser.heads import fit_parametric_head, l2_normalize
from fiser.inference import _load_archive


def build_head(
    archive_path: str | Path,
    output: str | Path,
    layer: int,
    kind: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> Path:
    archive = _load_archive(archive_path)
    if layer not in archive["embeddings"]:
        raise ValueError(f"Layer {layer} is not present in {archive_path}")
    features = archive["embeddings"][layer]
    features = features.float().numpy() if torch.is_tensor(features) else np.asarray(features, dtype=np.float32)
    labels = archive["binary_labels"].numpy().astype(np.int64)
    if len(features) != len(labels):
        raise ValueError("Embedding and binary-label lengths differ")
    if max_samples is not None and int(max_samples) < len(features):
        rng = np.random.default_rng(int(seed))
        selected = np.sort(rng.choice(len(features), size=int(max_samples), replace=False))
        features = features[selected]
        labels = labels[selected]
    features = l2_normalize(features)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("An inference head needs both natural (0) and synthetic (1) samples")

    if kind == "prototype":
        prototypes = np.stack([features[labels == label].mean(axis=0) for label in (0, 1)])
        payload = {
            "head_type": "prototype",
            "layer": int(layer),
            "prototypes": torch.from_numpy(l2_normalize(prototypes)),
            "classes": ["natural", "synthetic"],
            "feature_dim": int(features.shape[1]),
            "num_samples": int(len(features)),
            "metadata": {"archive": str(archive_path), "normalized": True},
        }
    elif kind == "linear":
        model = fit_parametric_head("linear", features, labels, seed=seed)
        payload = {
            "head_type": "linear",
            "layer": int(layer),
            "weight": torch.from_numpy(np.asarray(model.coef_, dtype=np.float32)),
            "bias": torch.from_numpy(np.asarray(model.intercept_, dtype=np.float32)),
            "classes": ["natural", "synthetic"],
            "feature_dim": int(features.shape[1]),
            "num_samples": int(len(features)),
            "metadata": {"archive": str(archive_path), "normalized": True, "estimator": "sklearn.LogisticRegression"},
        }
    else:
        raise ValueError("kind must be prototype or linear")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable FiSeR single-image head")
    parser.add_argument("--archive", required=True, help="Embedding archive used as the training feature bank")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--kind", choices=["prototype", "linear"], default="linear")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_head(
            archive_path=args.archive,
            output=args.output,
            layer=args.layer,
            kind=args.kind,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    )
