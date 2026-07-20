from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have KEY=VALUE form: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override has an empty key: {item}")
        result[key] = yaml.safe_load(raw_value)
    return result


def require_keys(config: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if config.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")
