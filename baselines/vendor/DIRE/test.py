from utils.config import cfg  # isort: split

import csv
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.eval import get_val_cfg, validate
from utils.utils import get_network
from test_logger import log_test_run


def save_metrics_json(path, payload):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

cfg = get_val_cfg(cfg, split="test", copy=False)

assert cfg.ckpt_path, "Please specify the path to the model checkpoint"
model_name = os.path.basename(cfg.ckpt_path).replace(".pth", "")
dataset_root = cfg.dataset_root  # keep it
rows = []
test_lmdb = cfg.lmdb_path
eval_names = cfg.eval_names
display_name = cfg.exp_name or Path(__file__).resolve().parent.name

model = get_network(cfg.arch)
state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
model_state = state_dict.get("model", state_dict)
if any(key.startswith("module.") for key in model_state.keys()):
    model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
model.load_state_dict(model_state, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

if test_lmdb:
    lmdb_paths = [p.strip() for p in test_lmdb.split(",") if p.strip()]
    names = [n.strip() for n in eval_names.split(",")] if eval_names else lmdb_paths
    if len(names) != len(lmdb_paths):
        raise ValueError("Length of eval_names must match lmdb_path when provided.")
    tqdm.write(f"'{display_name}:{model_name}' model testing on LMDB...")
    for i, (name, path) in enumerate(zip(names, lmdb_paths)):
        dataloader = create_dataloader(cfg, lmdb_path=path, is_train=False, distributed=False)
        metrics, _, _ = validate(
            model,
            dataloader,
            device=device,
            distributed=False,
            world_size=1,
            rank=0,
            desc=f"Test {name}",
        )
        tqdm.write(f"{name}:")
        for k, v in metrics.items():
            tqdm.write(f"{k}: {v:.5f}")
        tqdm.write("*" * 50)
        log_test_run(
            project_root=Path(__file__).resolve().parent,
            project_name=Path(__file__).resolve().parent.name,
            dataset_path=path,
            ckpt_path=cfg.ckpt_path,
            metrics=metrics,
            cmd=" ".join(sys.argv),
        )
        save_metrics_json(
            getattr(cfg, "metrics_out", ""),
            {
                "project": Path(__file__).resolve().parent.name,
                "dataset_path": path,
                "ckpt_path": cfg.ckpt_path,
                "corruption_type": getattr(cfg, "corruption_type", "none"),
                "corruption_value": getattr(cfg, "corruption_value", 0.0),
                "crop_mode": getattr(cfg, "crop_mode", "center"),
                "metrics": metrics,
            },
        )
        if i == 0:
            rows.append(["TestSet"] + list(metrics.keys()))
        rows.append([name] + list(metrics.values()))
else:
    tqdm.write(f"'{display_name}:{model_name}' model testing on folder datasets...")
    for i, dataset in enumerate(cfg.datasets_test):
        cfg.dataset_root = os.path.join(dataset_root, dataset)
        cfg.datasets = [""]
        dataloader = create_dataloader(cfg, lmdb_path=None, is_train=False, distributed=False)
        metrics, _, _ = validate(
            model,
            dataloader,
            device=device,
            distributed=False,
            world_size=1,
            rank=0,
            desc=f"Test {dataset}",
        )
        tqdm.write(f"{dataset}:")
        for k, v in metrics.items():
            tqdm.write(f"{k}: {v:.5f}")
        tqdm.write("*" * 50)
        log_test_run(
            project_root=Path(__file__).resolve().parent,
            project_name=Path(__file__).resolve().parent.name,
            dataset_path=cfg.dataset_root,
            ckpt_path=cfg.ckpt_path,
            metrics=metrics,
            cmd=" ".join(sys.argv),
        )
        if i == 0:
            rows.append(["TestSet"] + list(metrics.keys()))
        rows.append([dataset] + list(metrics.values()))

results_dir = os.path.join(cfg.root_dir, "data", "results")
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, f"{display_name}-{model_name}.csv"), "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerows(rows)
