import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lmdb_dataset import LmdbSampleDataset, arbitrary_resolution_collate, use_arbitrary_resolution_eval
from runner_utils import (
    AverageMeter,
    cleanup_distributed,
    compute_binary_metrics,
    create_config,
    gather_predictions,
    get_autocast_context,
    init_distributed,
    is_main_process,
    normalize_precision_mode,
    reduce_mean,
    setup_logger,
    unwrap_model,
)
from spai.models import build_cls_model
from spai.models.losses import build_loss


def parse_args():
    parser = argparse.ArgumentParser(description="SPAI LMDB testing")
    parser.add_argument("--cfg", type=Path, default=PROJECT_ROOT / "configs" / "spai.yaml")
    parser.add_argument("--lmdb_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "wildfake_spai")
    parser.add_argument("--name", type=str, default="wildfake_spai")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16", "bf16-mixed", "fp16", "fp16-mixed", "fp32"],
    )
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--feature_extraction_batch", type=int, default=None)
    parser.add_argument("--opt", nargs=2, action="append", default=[])
    return parser.parse_args()


def load_model_checkpoint(model, checkpoint_path: Path, logger) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    incompatible = unwrap_model(model).load_state_dict(state_dict, strict=True)
    logger.info("Loaded model checkpoint from %s", checkpoint_path)
    logger.info("Missing keys: %d | Unexpected keys: %d", len(incompatible.missing_keys), len(incompatible.unexpected_keys))


@torch.inference_mode()
def evaluate(config, model, criterion, data_loader, precision: str, logger):
    model.eval()
    criterion.eval()

    progress = tqdm(
        data_loader,
        total=len(data_loader),
        desc="Testing",
        disable=not is_main_process(),
        dynamic_ncols=True,
    )

    loss_sum = torch.zeros(1, device="cuda")
    loss_count = torch.zeros(1, device="cuda")
    loss_meter = AverageMeter()
    all_indices = []
    all_targets = []
    all_scores = []

    for step, (images, targets, indices) in enumerate(progress):
        targets = targets.cuda(non_blocking=True).view(-1)
        indices_cpu = indices.cpu().tolist()

        with get_autocast_context(precision):
            if isinstance(images, list):
                images = [image.cuda(non_blocking=True) for image in images]
                logits = model(images, config.MODEL.FEATURE_EXTRACTION_BATCH).squeeze(dim=1)
            else:
                images = images.cuda(non_blocking=True)
                logits = model(images).squeeze(dim=1)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits).float().cpu().tolist()
        reduced_loss = reduce_mean(loss.detach())
        loss_meter.update(reduced_loss.item(), targets.size(0))
        loss_sum += loss.detach() * targets.size(0)
        loss_count += targets.size(0)

        all_indices.extend(indices_cpu)
        all_targets.extend(targets.float().cpu().tolist())
        all_scores.extend(probs)

        if is_main_process():
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if is_main_process() and ((step + 1) % config.PRINT_FREQ == 0 or (step + 1) == len(data_loader)):
            logger.info("Step %d/%d | loss %.4f", step + 1, len(data_loader), loss_meter.avg)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(loss_count, op=torch.distributed.ReduceOp.SUM)

    _, gathered_targets, gathered_scores = gather_predictions(all_indices, all_targets, all_scores)
    metrics = compute_binary_metrics(gathered_targets, gathered_scores)
    metrics["loss"] = float((loss_sum / loss_count.clamp_min(1)).item())
    return metrics


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    torch.set_float32_matmul_precision("high")

    args.test_batch_size = args.batch_size
    config = create_config(args)
    dynamic_resolution_eval = use_arbitrary_resolution_eval(config, is_train=False)
    cudnn.benchmark = not dynamic_resolution_eval
    output_dir = Path(config.OUTPUT)
    logger = setup_logger(output_dir, "test.log")
    precision_mode = normalize_precision_mode(args.precision)

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w") as f:
            f.write(config.dump())
        logger.info("Testing LMDB: %s", args.lmdb_path)
        logger.info("Model checkpoint: %s", args.model_path)
        logger.info(
            "Precision request: %s | effective autocast dtype: %s",
            args.precision,
            precision_mode,
        )
        logger.info(
            "Dynamic resolution eval: %s | cudnn.benchmark=%s | feature_extraction_batch=%s",
            dynamic_resolution_eval,
            cudnn.benchmark,
            config.MODEL.FEATURE_EXTRACTION_BATCH,
        )

    dataset = LmdbSampleDataset(str(args.lmdb_path), config=config, is_train=False)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    collate_fn = arbitrary_resolution_collate if use_arbitrary_resolution_eval(config, is_train=False) else None
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=config.DATA.NUM_WORKERS > 0,
        prefetch_factor=config.DATA.TEST_PREFETCH_FACTOR if config.DATA.NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
    )

    model = build_cls_model(config).cuda()
    load_model_checkpoint(model, args.model_path, logger)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    criterion = build_loss(config).cuda()

    metrics = evaluate(
        config=config,
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        precision=precision_mode,
        logger=logger,
    )

    if is_main_process():
        logger.info(
            "Results | loss %.4f | ACC %.4f | AP %.4f | PR-AUC %.4f | AUROC %.4f | AvgRecall %.4f | TPR@0.05FPR %.4f",
            metrics["loss"],
            metrics["acc"],
            metrics["ap"],
            metrics["pr_auc"],
            metrics["auroc"],
            metrics["avg_recall"],
            metrics["tpr_at_fpr"],
        )
        metrics_path = output_dir / f"{args.model_path.stem}__{args.lmdb_path.name}.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", metrics_path)

    cleanup_distributed()


if __name__ == "__main__":
    main()
