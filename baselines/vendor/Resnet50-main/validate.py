import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

from utils.metric_utils import evaluate_metrics


def _gather_predictions(y_true, y_pred, device, world_size):
    true_tensor = torch.tensor(y_true, device=device, dtype=torch.float32)
    pred_tensor = torch.tensor(y_pred, device=device, dtype=torch.float32)

    local_len = torch.tensor([len(true_tensor)], device=device, dtype=torch.long)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len)
    max_len = int(torch.stack(gathered_lens).max().item())

    def pad_tensor(tensor, target_len, pad_val):
        if tensor.numel() < target_len:
            pad_size = target_len - tensor.numel()
            tensor = torch.cat([tensor, torch.full((pad_size,), pad_val, device=tensor.device)])
        return tensor

    true_tensor = pad_tensor(true_tensor, max_len, -1)
    pred_tensor = pad_tensor(pred_tensor, max_len, -1)

    gathered_true = [torch.zeros_like(true_tensor) for _ in range(world_size)]
    gathered_pred = [torch.zeros_like(pred_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_true, true_tensor)
    dist.all_gather(gathered_pred, pred_tensor)

    merged_true = torch.cat(gathered_true).cpu().numpy()
    merged_pred = torch.cat(gathered_pred).cpu().numpy()
    mask = merged_true >= 0
    return merged_true[mask], merged_pred[mask]


def validate(model, data_loader, device=None, distributed=False, world_size=1, rank=0, desc="Validating"):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    with torch.no_grad():
        y_true, y_pred = [], []
        show_progress = (not distributed) or rank == 0
        progress_bar = tqdm(total=len(data_loader), desc=desc, dynamic_ncols=True, disable=not show_progress)

        for img, label in data_loader:
            in_tens = img.to(device)
            preds = model(in_tens).sigmoid().flatten()
            y_pred.extend(preds.detach().cpu().tolist())
            y_true.extend(label.flatten().tolist())

            progress_bar.update(1)

        progress_bar.close()

    if distributed:
        y_true, y_pred = _gather_predictions(y_true, y_pred, device, world_size)
    else:
        y_true, y_pred = np.array(y_true), np.array(y_pred)

    metrics = evaluate_metrics(y_true, y_pred)
    return metrics, y_true, y_pred
