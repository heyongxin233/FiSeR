import ast
import datetime
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def parse_override_value(raw_value: str):
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        torch.cuda.set_device(0)
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
        dist.destroy_process_group()


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return get_rank() == 0


def get_rank() -> int:
    if not is_dist_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    if is_dist_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def set_seed(seed: int) -> None:
    rank = get_rank()
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: Path, filename: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger_name = f"spai_runner_rank_{get_rank()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s][rank %(rank)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = get_rank()
            return True

    if is_main_process():
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        console_handler.addFilter(RankFilter())
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / filename, mode="a")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(RankFilter())
    logger.addHandler(file_handler)
    return logger


def create_config(args) -> "CfgNode":
    from spai.config import get_custom_config

    config = get_custom_config(str(args.cfg))
    config.defrost()

    if getattr(args, "batch_size", None) is not None:
        config.DATA.BATCH_SIZE = args.batch_size
    if getattr(args, "epochs", None) is not None:
        config.TRAIN.EPOCHS = args.epochs
    if getattr(args, "learning_rate", None) is not None:
        config.TRAIN.BASE_LR = args.learning_rate
    if getattr(args, "weight_decay", None) is not None:
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
    if getattr(args, "num_workers", None) is not None:
        config.DATA.NUM_WORKERS = args.num_workers
    if getattr(args, "img_size", None) is not None:
        config.DATA.IMG_SIZE = args.img_size
    if getattr(args, "feature_extraction_batch", None) is not None:
        config.MODEL.FEATURE_EXTRACTION_BATCH = args.feature_extraction_batch
    if getattr(args, "print_freq", None) is not None:
        config.PRINT_FREQ = args.print_freq
    if getattr(args, "save_every", None) is not None:
        config.SAVE_FREQ = args.save_every

    config.PRETRAINED = str(args.pretrained) if getattr(args, "pretrained", None) else ""
    config.AMP_OPT_LEVEL = "O0"
    config.OUTPUT = str(Path(args.output).resolve())
    config.TAG = getattr(args, "name", "spai")
    config.DATA.PIN_MEMORY = True
    config.DATA.VAL_BATCH_SIZE = getattr(args, "test_batch_size", None)
    config.DATA.TEST_BATCH_SIZE = getattr(args, "test_batch_size", None)

    if getattr(args, "opt", None):
        overrides = []
        for key, value in args.opt:
            overrides.extend([key, parse_override_value(value)])
        config.merge_from_list(overrides)

    config.freeze()
    return config


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def auto_resume_path(output_dir: Path) -> Optional[Path]:
    checkpoints = sorted(output_dir.glob("ckpt_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return None
    return checkpoints[-1]


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config.dump(),
    }
    save_path = output_dir / f"ckpt_epoch_{epoch}.pth"
    torch.save(checkpoint, save_path)
    return save_path


def load_resume_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("lr_scheduler") is not None:
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", -1)) + 1


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: Path, logger: logging.Logger) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    if any(key.startswith("encoder.") for key in checkpoint_model.keys()):
        checkpoint_model = {
            key.replace("encoder.", ""): value
            for key, value in checkpoint_model.items()
            if key.startswith("encoder.")
        }
        logger.info("Detected MFM pretrain checkpoint, stripped `encoder.` prefix.")

    model_state = unwrap_model(model).state_dict()
    filtered_state = {}
    skipped_keys = []
    for key, value in checkpoint_model.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            skipped_keys.append(key)

    incompatible = unwrap_model(model).load_state_dict(filtered_state, strict=False)
    logger.info("Loaded pretrained weights from %s", checkpoint_path)
    logger.info("Missing keys: %d | Unexpected keys: %d", len(incompatible.missing_keys), len(incompatible.unexpected_keys))
    logger.info("Skipped incompatible pretrained keys: %d", len(skipped_keys))


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_dist_initialized():
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced


def gather_predictions(
    indices: list[int],
    targets: list[float],
    scores: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = {
        "indices": np.asarray(indices, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.float32),
        "scores": np.asarray(scores, dtype=np.float32),
    }

    if not is_dist_initialized():
        return payload["indices"], payload["targets"], payload["scores"]

    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, payload)

    merged: dict[int, tuple[float, float]] = {}
    for item in gathered:
        for idx, target, score in zip(item["indices"], item["targets"], item["scores"]):
            merged[int(idx)] = (float(target), float(score))

    sorted_indices = np.asarray(sorted(merged.keys()), dtype=np.int64)
    sorted_targets = np.asarray([merged[idx][0] for idx in sorted_indices], dtype=np.float32)
    sorted_scores = np.asarray([merged[idx][1] for idx in sorted_indices], dtype=np.float32)
    return sorted_indices, sorted_targets, sorted_scores


def compute_binary_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    targets = targets.astype(np.int64)
    scores = scores.astype(np.float32)
    predictions = (scores >= 0.5).astype(np.int64)
    metrics = {
        "acc": float(accuracy_score(targets.astype(np.int64), predictions)),
        "ap": float("nan"),
        "auc": float("nan"),
        "pr_auc": float("nan"),
        "auroc": float("nan"),
        "avg_recall": float("nan"),
        "real_acc": float("nan"),
        "fake_acc": float("nan"),
        "tpr_at_fpr": float("nan"),
        "tpr_at_fpr_threshold": float("nan"),
    }
    unique_targets = np.unique(targets)
    real_mask = targets == 1
    fake_mask = targets == 0
    if real_mask.any():
        metrics["real_acc"] = float(accuracy_score(targets[real_mask], predictions[real_mask]))
    if fake_mask.any():
        metrics["fake_acc"] = float(accuracy_score(targets[fake_mask], predictions[fake_mask]))
    if not np.isnan(metrics["real_acc"]) and not np.isnan(metrics["fake_acc"]):
        metrics["avg_recall"] = (metrics["real_acc"] + metrics["fake_acc"]) / 2.0
    if unique_targets.size > 1:
        metrics["ap"] = float(average_precision_score(targets, scores))
        precision, recall, _ = precision_recall_curve(targets, scores, pos_label=1)
        metrics["pr_auc"] = float(auc(recall, precision))
        metrics["auc"] = float(roc_auc_score(targets, scores))
        metrics["auroc"] = metrics["auc"]
        fpr, tpr, thresholds = roc_curve(targets, scores)
        target_index = int(np.argmin(np.abs(fpr - 0.05)))
        metrics["tpr_at_fpr"] = float(tpr[target_index])
        metrics["tpr_at_fpr_threshold"] = float(thresholds[target_index])
    return metrics


def normalize_precision_mode(precision: str) -> str:
    precision = precision.lower()
    aliases = {
        "bf16-mixed": "bf16",
        "fp16-mixed": "fp16",
    }
    return aliases.get(precision, precision)


def get_autocast_context(precision: str):
    precision = normalize_precision_mode(precision)
    if precision == "fp32":
        return torch.autocast(device_type="cuda", enabled=False)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported precision: {precision}")


def create_grad_scaler(precision: str) -> Optional[torch.cuda.amp.GradScaler]:
    if normalize_precision_mode(precision) == "fp16":
        return torch.cuda.amp.GradScaler()
    return None
