import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from data import create_dataloader
from networks.trainer import CLIPModelLora
from options.test_options import TestOptions
from test_logger import log_test_run
from validate import validate


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_distributed(opt):
    env_rank = os.environ.get("RANK")
    env_world = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")

    if env_rank is not None and env_world is not None:
        opt.rank = int(env_rank)
        opt.world_size = int(env_world)
        opt.local_rank = int(local_rank_env) if local_rank_env is not None else 0
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(opt.local_rank)
        opt.distributed = True
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.local_rank = 0
        opt.distributed = False

    opt.device = torch.device("cuda", opt.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    opt.gpu_ids = [opt.local_rank] if torch.cuda.is_available() else []
    return opt


def clean_state_dict(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


if __name__ == "__main__":
    seed_torch(100)
    opt = TestOptions().parse(print_options=False)
    opt = init_distributed(opt)

    if not opt.lmdb_path:
        raise ValueError("Please provide --lmdb_path pointing to one or more LMDB test sets.")

    lmdb_paths = [item.strip() for item in opt.lmdb_path.split(",") if item.strip()]
    eval_names = [item.strip() for item in opt.eval_names.split(",")] if opt.eval_names else lmdb_paths
    if len(eval_names) != len(lmdb_paths):
        raise ValueError("Length of eval_names must match lmdb_path.")

    if opt.rank == 0:
        print(f"Model_path {opt.model_path}")

    checkpoint = torch.load(opt.model_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = clean_state_dict(state_dict)

    model = CLIPModelLora(
        name=opt.clip,
        lora_r=opt.lora_r,
        lora_alpha=opt.lora_alpha,
        lora_dropout=opt.lora_dropout,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(opt.device)

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
        )
    model.eval()

    for eval_name, lmdb_path in zip(eval_names, lmdb_paths):
        dataloader, sampler = create_dataloader(
            opt,
            lmdb_path=lmdb_path,
            is_train=False,
            distributed=opt.distributed,
        )
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)

        metrics, _, _ = validate(
            model,
            dataloader,
            device=opt.device,
            distributed=opt.distributed,
            world_size=opt.world_size,
            rank=opt.rank,
            desc=f"Testing {eval_name}",
            amp=opt.amp,
            amp_dtype=opt.amp_dtype,
        )

        if opt.rank == 0 and metrics is not None:
            print(
                f"({eval_name}) acc: {metrics['acc'] * 100:.2f}; "
                f"pr_auc: {metrics['pr_auc'] * 100:.2f}; "
                f"auroc: {metrics['auroc'] * 100:.2f}; "
                f"avg_recall: {metrics['avg_recall'] * 100:.2f}; "
                f"tpr@0.05fpr: {metrics['tpr_at_fpr'] * 100:.2f}"
            )
            log_test_run(
                project_root=Path(__file__).resolve().parent,
                project_name=Path(__file__).resolve().parent.name,
                dataset_path=lmdb_path,
                ckpt_path=opt.model_path,
                metrics=metrics,
                cmd=" ".join(sys.argv),
            )

    if opt.distributed:
        dist.destroy_process_group()
