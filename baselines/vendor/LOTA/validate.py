from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from metric_utils import evaluate_metrics


def _gather_predictions(y_true, y_pred, device, world_size):
    true_tensor = torch.tensor(y_true, device=device, dtype=torch.float32)
    pred_tensor = torch.tensor(y_pred, device=device, dtype=torch.float32)

    local_len = torch.tensor([len(true_tensor)], device=device, dtype=torch.long)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len)
    max_len = int(torch.stack(gathered_lens).max().item())

    def pad_tensor(tensor, target_len, pad_value):
        if tensor.numel() < target_len:
            padding = torch.full((target_len - tensor.numel(),), pad_value, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=0)
        return tensor

    true_tensor = pad_tensor(true_tensor, max_len, -1)
    pred_tensor = pad_tensor(pred_tensor, max_len, -1)

    gathered_true = [torch.zeros_like(true_tensor) for _ in range(world_size)]
    gathered_pred = [torch.zeros_like(pred_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_true, true_tensor)
    dist.all_gather(gathered_pred, pred_tensor)

    merged_true = torch.cat(gathered_true).cpu().numpy()
    merged_pred = torch.cat(gathered_pred).cpu().numpy()
    valid_mask = merged_true >= 0
    return merged_true[valid_mask], merged_pred[valid_mask]


def get_amp_context(precision, device):
    if device.type != 'cuda' or precision == 'fp32':
        return nullcontext()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)


def validate(
    model,
    data_loader,
    device,
    distributed=False,
    world_size=1,
    rank=0,
    desc='Evaluating',
    precision='bf16',
    max_batches=0,
):
    model.eval()
    y_true = []
    y_pred = []
    show_progress = (not distributed) or rank == 0

    with torch.no_grad():
        progress = tqdm(
            total=len(data_loader),
            desc=desc,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for batch_index, (images, labels) in enumerate(data_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1)

            with get_amp_context(precision, device):
                logits = model(images).view(-1)
            probs = torch.sigmoid(logits.float())

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(probs.cpu().numpy().tolist())
            progress.update(1)

            if max_batches > 0 and batch_index >= max_batches:
                break
        progress.close()

    if distributed:
        y_true, y_pred = _gather_predictions(y_true, y_pred, device, world_size)
    else:
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

    metrics = evaluate_metrics(y_true, y_pred)
    return metrics, y_true, y_pred
