import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from utils.config import CONFIGCLASS
from utils.metric_utils import evaluate_metrics
from utils.utils import to_cuda


def get_val_cfg(cfg: CONFIGCLASS, split="val", copy=True):
    if copy:
        from copy import deepcopy

        val_cfg = deepcopy(cfg)
    else:
        val_cfg = cfg
    val_cfg.dataset_root = os.path.join(val_cfg.dataset_root, split)
    val_cfg.datasets = cfg.datasets_test
    val_cfg.isTrain = False
    # val_cfg.aug_resize = False
    # val_cfg.aug_crop = False
    val_cfg.aug_flip = False
    val_cfg.serial_batches = True
    val_cfg.jpg_method = ["pil"]
    # Currently assumes jpg_prob, blur_prob 0 or 1
    if len(val_cfg.blur_sig) == 2:
        b_sig = val_cfg.blur_sig
        val_cfg.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_cfg.jpg_qual) != 1:
        j_qual = val_cfg.jpg_qual
        val_cfg.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_cfg

def _gather_tensor(tensor, world_size, pad_value=-1):
    local_len = torch.tensor([tensor.numel()], device=tensor.device, dtype=torch.long)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len)
    max_len = int(torch.stack(gathered_lens).max().item())

    if tensor.numel() < max_len:
        pad_size = max_len - tensor.numel()
        tensor = torch.cat([tensor, torch.full((pad_size,), pad_value, device=tensor.device)])

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)


def validate(model: nn.Module, data_loader, device=None, distributed=False, world_size=1, rank=0, desc="Validating"):
    if data_loader is None:
        return None, None, None

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        y_true, y_pred = [], []
        show_progress = (not distributed) or rank == 0
        progress_bar = tqdm(total=len(data_loader), desc=desc, dynamic_ncols=True) if show_progress else None

        for data in data_loader:
            img, label, meta = data if len(data) == 3 else (*data, None)
            in_tens = to_cuda(img, device)
            meta = to_cuda(meta, device)
            predict = model(in_tens, meta).sigmoid().flatten()
            y_pred.extend(predict.detach().cpu().tolist())
            y_true.extend(label.flatten().tolist())

            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

    if distributed:
        true_tensor = torch.tensor(y_true, device=device, dtype=torch.float32)
        pred_tensor = torch.tensor(y_pred, device=device, dtype=torch.float32)
        true_tensor = _gather_tensor(true_tensor, world_size)
        pred_tensor = _gather_tensor(pred_tensor, world_size)
        mask = true_tensor >= 0
        y_true = true_tensor[mask].cpu().numpy()
        y_pred = pred_tensor[mask].cpu().numpy()
    else:
        y_true, y_pred = np.array(y_true), np.array(y_pred)

    metrics = evaluate_metrics(y_true, y_pred)
    return metrics, y_true, y_pred
