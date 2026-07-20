import random

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from data import create_dataloader
from models import get_model
from options.test_options import TestOptions


SEED = 0


def is_main_process(opt):
    return (not opt.distributed) or opt.local_rank == 0


def setup_distributed(opt):
    if opt.distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def cleanup_distributed(opt):
    if opt.distributed and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(rank=0):
    seed = SEED + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def gather_numpy_array(array, opt):
    if not opt.distributed:
        return array

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, array)
    arrays = [np.asarray(item) for item in gathered if item is not None and len(item) > 0]
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def evaluate_metrics(y_true, y_score, target_fpr=0.05):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)

    if y_true.size == 0:
        raise ValueError("No predictions were collected for evaluation.")

    epsilon = 1e-6
    ap = average_precision_score(y_true, y_score)

    if len(np.unique(y_true)) < 2:
        threshold = 0.5
        y_pred = (y_score >= threshold).astype(int)
        real_mask = y_true == 1
        fake_mask = y_true == 0
        real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask]) if real_mask.any() else 0.0
        fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask]) if fake_mask.any() else 0.0
        acc = accuracy_score(y_true, y_pred)
        avg_recall = (real_acc + fake_acc) / 2
        return {
            "ap": float(ap),
            "pr_auc": float("nan"),
            "auroc": float("nan"),
            "F1": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "threshold": float(threshold),
            "acc": float(acc),
            "avg_recall": float(avg_recall),
            "real_acc": float(real_acc),
            "fake_acc": float(fake_acc),
            "tpr_at_fpr": float("nan"),
            "tpr_at_fpr_threshold": float("nan"),
        }

    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=1)
    pr_auc = auc(recall, precision)
    auroc = roc_auc_score(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + epsilon)
    best_index = int(f1_scores.argmax())
    threshold = float(thresholds[best_index]) if best_index < len(thresholds) else 1.0

    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    real_mask = y_true == 1
    fake_mask = y_true == 0
    real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask]) if real_mask.any() else 0.0
    fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask]) if fake_mask.any() else 0.0
    avg_recall = (real_acc + fake_acc) / 2

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    target_index = int(np.argmin(np.abs(fpr - target_fpr)))

    return {
        "ap": float(ap),
        "pr_auc": float(pr_auc),
        "auroc": float(auroc),
        "F1": float(f1_scores[best_index]),
        "Precision": float(precision[best_index]),
        "Recall": float(recall[best_index]),
        "threshold": threshold,
        "acc": float(acc),
        "avg_recall": float(avg_recall),
        "real_acc": float(real_acc),
        "fake_acc": float(fake_acc),
        "tpr_at_fpr": float(tpr[target_index]),
        "tpr_at_fpr_threshold": float(roc_thresholds[target_index]),
    }


def load_checkpoint(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
        cleaned_state_dict[key] = value
    state_dict = cleaned_state_dict
    model.load_state_dict(state_dict, strict=True)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    prediction_chunks = []
    label_chunks = []
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).view(-1)
        prediction_chunks.append(torch.sigmoid(logits).cpu())
        label_chunks.append(labels.cpu().float())

    if not prediction_chunks:
        return np.array([]), np.array([])

    return torch.cat(prediction_chunks).numpy(), torch.cat(label_chunks).numpy()


def main():
    opt = TestOptions().parse(print_options=False)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    if not opt.lmdb_path:
        raise ValueError("--lmdb_path is required for LMDB evaluation.")
    if not opt.model_path:
        raise ValueError("--model_path is required for checkpoint loading.")

    setup_distributed(opt)
    try:
        set_seed(opt.local_rank if opt.distributed else 0)

        device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")
        data_loader = create_dataloader(opt)
        model = get_model(opt.arch, opt).to(device)
        load_checkpoint(model, opt.model_path)
        if opt.compile and hasattr(torch, "compile"):
            model = torch.compile(
                model,
                backend=opt.compile_backend,
                mode=opt.compile_mode,
            )

        y_pred, y_true = evaluate(model, data_loader, device)
        y_pred = gather_numpy_array(y_pred, opt)
        y_true = gather_numpy_array(y_true, opt)

        if not is_main_process(opt):
            return

        metrics = evaluate_metrics(y_true, y_pred)
        print(f"LMDB: {opt.lmdb_path}")
        print(f"Checkpoint: {opt.model_path}")
        print(f"AP: {metrics['ap'] * 100:.2f}")
        print(f"PR-AUC: {metrics['pr_auc'] * 100:.2f}")
        print(f"AUROC: {metrics['auroc'] * 100:.2f}")
        print(f"Real Acc: {metrics['real_acc'] * 100:.2f}")
        print(f"Fake Acc: {metrics['fake_acc'] * 100:.2f}")
        print(f"Avg Recall: {metrics['avg_recall'] * 100:.2f}")
        print(f"Acc: {metrics['acc'] * 100:.2f}")
        print(f"TPR@0.05FPR: {metrics['tpr_at_fpr'] * 100:.2f}")
        print(f"Threshold: {metrics['threshold']:.6f}")
    finally:
        cleanup_distributed(opt)


if __name__ == "__main__":
    main()
