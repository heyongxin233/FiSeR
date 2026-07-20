import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

from metric_utils import evaluate_metrics


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
    merged = torch.cat(gathered)
    return merged


def validate(model, val_loader, device=None, distributed=False, world_size=1, rank=0, desc="Validating"):
    if val_loader is None:
        return None, None, None

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        y_true, y_pred = [], []
        show_progress = (not distributed) or rank == 0
        progress_bar = tqdm(total=len(val_loader), desc=desc, dynamic_ncols=True) if show_progress else None

        for img, label in val_loader:
            in_tens = img.to(device, non_blocking=True)
            preds = model(in_tens).sigmoid().flatten()
            y_pred.extend(preds.detach().cpu().tolist())
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
