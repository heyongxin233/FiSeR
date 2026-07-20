from __future__ import annotations

import json
import numbers
import os
from datetime import datetime
from pathlib import Path


def _format_metric_value(value):
    if value is None:
        return "unknown"
    if isinstance(value, numbers.Number):
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)
    return str(value)


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def log_test_run(project_root, project_name, dataset_path, ckpt_path, metrics, cmd=None, extra=None):
    try:
        configured_dir = os.environ.get("BASELINE_LOG_DIR")
        logs_dir = Path(configured_dir).expanduser() if configured_dir else Path(project_root) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        time_str = datetime.now().isoformat(timespec="seconds")
        dataset_value = dataset_path if dataset_path else "unknown"
        ckpt_value = ckpt_path if ckpt_path else "unknown"
        cmd_value = cmd if cmd else "unknown"
        metrics = metrics or {}

        lines = [
            "=" * 60,
            f"time: {time_str}",
            f"project: {project_name}",
            f"cmd: {cmd_value}",
            "dataset:",
            f"  {dataset_value}",
            "ckpt:",
            f"  {ckpt_value}",
            "metrics:",
        ]
        for key, value in metrics.items():
            lines.append(f"  {key}: {_format_metric_value(value)}")
        lines.append("=" * 60)
        lines.append("")

        with (logs_dir / "logs.txt").open("a", encoding="utf-8") as file:
            file.write("\n".join(lines))

        payload = {
            "time": time_str,
            "project": project_name,
            "dataset_path": dataset_value,
            "ckpt_path": ckpt_value,
            "metrics": {k: _json_safe(v) for k, v in metrics.items()},
        }
        if cmd:
            payload["cmd"] = cmd
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is not None:
            payload["CUDA_VISIBLE_DEVICES"] = cuda_visible
        if extra:
            payload.update({k: _json_safe(v) for k, v in extra.items()})

        with (logs_dir / "logs.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
