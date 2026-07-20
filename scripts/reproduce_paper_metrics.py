from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate import evaluate_knn, load_archive, parse_devices
from fiser.config import load_yaml
from fiser.faiss_utils import import_faiss_gpu


def parse_archive_args(values: list[str]) -> dict[str, Path]:
    archives: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Archive must have NAME=PATH form: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        if not name or name in archives:
            raise ValueError(f"Archive name is empty or duplicated: {value}")
        archives[name] = Path(raw_path).expanduser()
    return archives


def _metric_check(actual: float, expected: float, tolerance: float) -> dict[str, Any]:
    delta = float(actual) - float(expected)
    return {
        "actual": float(actual),
        "expected": float(expected),
        "delta": delta,
        "passed": abs(delta) <= float(tolerance),
    }


def reproduce(
    config: dict[str, Any],
    database_path: str | Path,
    archives: dict[str, Path],
    devices: list[int] | None,
    faiss_fp16: bool = False,
) -> dict[str, Any]:
    import_faiss_gpu()
    dataset_config = config.get("datasets")
    if not isinstance(dataset_config, dict) or not dataset_config:
        raise ValueError("Paper metric config has no datasets")
    missing = sorted(set(dataset_config) - set(archives))
    extra = sorted(set(archives) - set(dataset_config))
    if missing or extra:
        raise ValueError(f"Archive names mismatch: missing={missing}, extra={extra}")

    database = load_archive(database_path)
    tolerance = float(config.get("tolerance", 5e-6))
    rows: list[dict[str, Any]] = []
    passed = True
    for name, settings in dataset_config.items():
        query = load_archive(archives[name])
        layer = int(settings["layer"])
        result = evaluate_knn(
            database,
            query,
            max_k=int(config.get("max_k", 51)),
            temperature=float(config.get("temperature", 0.05)),
            devices=devices,
            add_batch_size=65536,
            query_batch_size=8192,
            protocol=str(config.get("protocol", "legacy")),
            target_fpr=float(config.get("target_fpr", 0.05)),
            requested_layers=[layer],
            faiss_fp16=faiss_fp16,
        )
        actual = result["best"]
        expected = settings["expected"]
        checks = {
            "k": {
                "actual": int(actual["k"]),
                "expected": int(expected["k"]),
                "passed": int(actual["k"]) == int(expected["k"]),
            },
            "auroc": _metric_check(actual["auroc"], expected["auroc"], tolerance),
            "tpr_at_fpr": _metric_check(
                actual["tpr_at_fpr"], expected["tpr_at_fpr"], tolerance
            ),
        }
        row_passed = all(check["passed"] for check in checks.values())
        passed = passed and row_passed
        rows.append(
            {
                "name": name,
                "display_name": str(settings.get("display_name", name)),
                "archive": str(archives[name]),
                "layer": layer,
                "k": int(actual["k"]),
                "auroc": float(actual["auroc"]),
                "tpr_at_fpr": float(actual["tpr_at_fpr"]),
                "checks": checks,
                "passed": row_passed,
            }
        )

    average_actual = {
        "auroc": float(np.mean([row["auroc"] for row in rows])),
        "tpr_at_fpr": float(np.mean([row["tpr_at_fpr"] for row in rows])),
    }
    average_expected = config["average"]["expected"]
    average_checks = {
        metric: _metric_check(value, average_expected[metric], tolerance)
        for metric, value in average_actual.items()
    }
    passed = passed and all(check["passed"] for check in average_checks.values())
    return {
        "passed": passed,
        "protocol": str(config.get("protocol", "legacy")),
        "database_archive": str(database_path),
        "tolerance": tolerance,
        "datasets": rows,
        "average": {**average_actual, "checks": average_checks},
    }


def print_table(result: dict[str, Any]) -> None:
    print("| Dataset | Layer | k | AUROC | TPR@5% FPR | Match |")
    print("| --- | ---: | ---: | ---: | ---: | :---: |")
    for row in result["datasets"]:
        marker = "yes" if row["passed"] else "no"
        print(
            f"| {row['display_name']} | {row['layer']} | {row['k']} | "
            f"{100 * row['auroc']:.4f} | {100 * row['tpr_at_fpr']:.4f} | {marker} |"
        )
    average = result["average"]
    marker = "yes" if result["passed"] else "no"
    print(
        f"| **Average** |  |  | **{100 * average['auroc']:.4f}** | "
        f"**{100 * average['tpr_at_fpr']:.4f}** | **{marker}** |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the FiSeR main table with GPU FAISS")
    parser.add_argument("--config", default="configs/paper_metrics.yaml")
    parser.add_argument("--database-pt", required=True)
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="One test archive; pass once for every dataset in the config",
    )
    parser.add_argument("--faiss-devices", default=None)
    parser.add_argument("--faiss-fp16", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = reproduce(
        load_yaml(args.config),
        database_path=args.database_pt,
        archives=parse_archive_args(args.archive),
        devices=parse_devices(args.faiss_devices),
        faiss_fp16=args.faiss_fp16,
    )
    print_table(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved reproduction report to {output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
