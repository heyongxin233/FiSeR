from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_registered_local_entrypoints_exist():
    registry = yaml.safe_load((ROOT / "baselines/registry.yaml").read_text(encoding="utf-8"))
    for name, entry in registry.items():
        if entry["kind"] == "local":
            for key in ("train_entrypoint", "eval_entrypoint", "config"):
                assert (ROOT / entry[key]).is_file(), f"{name}: missing {key}"
            continue
        if source_dir := entry.get("source_dir"):
            source = ROOT / source_dir
            assert source.is_dir(), f"{name}: missing source snapshot"
            assert (source / entry["train_entrypoint"]).exists(), f"{name}: missing train entrypoint"
            assert (source / entry["eval_entrypoint"]).exists(), f"{name}: missing eval entrypoint"
