import torch
from tqdm import tqdm

from metric_utils import evaluate_metrics


def validate(model, data_loader, device=None, distributed=False, world_size=1):
    if data_loader is None:
        return None

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        y_true, y_pred = [], []
        show_progress = (not distributed) or getattr(data_loader, "sampler", None) is None or getattr(data_loader.sampler, "rank", 0) == 0
        progress_bar = tqdm(total=len(data_loader), desc="Validating", dynamic_ncols=True) if show_progress else None

        for img, label in data_loader:
            in_tens = img.to(device, non_blocking=True)
            preds = model(in_tens).sigmoid().flatten()
            y_pred.extend(preds.detach().cpu().tolist())
            y_true.extend(label.flatten().tolist())

            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

    if distributed:
        y_true_tensor = torch.tensor(y_true, device=device, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, device=device, dtype=torch.float32)

        local_len = torch.tensor([len(y_true_tensor)], device=device, dtype=torch.long)
        gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_lens, local_len)
        max_len = int(torch.stack(gathered_lens).max().item())

        def pad_tensor(tensor, target_len, pad_val):
            if tensor.numel() < target_len:
                pad_size = target_len - tensor.numel()
                tensor = torch.cat([tensor, torch.full((pad_size,), pad_val, device=tensor.device)])
            return tensor

        y_true_tensor = pad_tensor(y_true_tensor, max_len, -1)
        y_pred_tensor = pad_tensor(y_pred_tensor, max_len, -1)

        gathered_true = [torch.zeros_like(y_true_tensor) for _ in range(world_size)]
        gathered_pred = [torch.zeros_like(y_pred_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_true, y_true_tensor)
        torch.distributed.all_gather(gathered_pred, y_pred_tensor)

        merged_true = torch.cat(gathered_true).cpu()
        merged_pred = torch.cat(gathered_pred).cpu()
        mask = merged_true >= 0
        y_true = merged_true[mask].numpy()
        y_pred = merged_pred[mask].numpy()

    metrics = evaluate_metrics(y_true, y_pred)
    return metrics
