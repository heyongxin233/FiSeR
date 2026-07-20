import argparse
import json
import os
import sys
import time
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

from lmdb_dataset import LmdbSampleDataset
from runner_utils import (
    AverageMeter,
    auto_resume_path,
    barrier,
    cleanup_distributed,
    create_config,
    create_grad_scaler,
    get_autocast_context,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    load_pretrained_weights,
    load_resume_checkpoint,
    reduce_mean,
    save_checkpoint,
    set_seed,
    setup_logger,
    unwrap_model,
)
from spai.lr_scheduler import build_scheduler
from spai.models import build_cls_model
from spai.models.losses import build_loss
from spai.optimizer import build_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="SPAI LMDB training")
    parser.add_argument("--cfg", type=Path, default=PROJECT_ROOT / "configs" / "spai.yaml")
    parser.add_argument("--train_lmdb", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ckpt" / "wildfake_spai")
    parser.add_argument("--name", type=str, default="wildfake_spai")
    parser.add_argument("--pretrained", type=Path, default=PROJECT_ROOT / "weights" / "mfm_pretrain_vit_base.pth")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--feature_extraction_batch", type=int, default=None)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--no_auto_resume", action="store_true")
    parser.add_argument("--opt", nargs=2, action="append", default=[])
    return parser.parse_args()


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    precision: str,
    logger,
):
    model.train()
    criterion.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = tqdm(
        data_loader,
        total=len(data_loader),
        desc=f"Epoch {epoch + 1}/{config.TRAIN.EPOCHS}",
        disable=not is_main_process(),
        dynamic_ncols=True,
    )

    optimizer.zero_grad(set_to_none=True)
    end = time.time()

    for step, (images, targets, _) in enumerate(progress):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).view(-1)

        with get_autocast_context(precision):
            logits = model(images).squeeze(dim=1)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.TRAIN.CLIP_GRAD is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step_update(epoch * len(data_loader) + step)

        reduced_loss = reduce_mean(loss.detach())
        loss_meter.update(reduced_loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process():
            progress.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
                time=f"{batch_time.avg:.3f}s",
            )

        if is_main_process() and ((step + 1) % config.PRINT_FREQ == 0 or (step + 1) == len(data_loader)):
            logger.info(
                "Epoch %d Step %d/%d | loss %.4f | lr %.6e",
                epoch,
                step + 1,
                len(data_loader),
                loss_meter.avg,
                optimizer.param_groups[-1]["lr"],
            )

    return {"loss": loss_meter.avg, "time": batch_time.avg}


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    torch.set_float32_matmul_precision("high")
    cudnn.benchmark = True

    config = create_config(args)
    output_dir = Path(config.OUTPUT)
    logger = setup_logger(output_dir, "train.log")
    set_seed(args.seed if args.seed is not None else config.SEED)

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w") as f:
            f.write(config.dump())
        logger.info("Using GPUs: world_size=%d local_rank=%d", world_size, local_rank)
        logger.info("Training LMDB: %s", args.train_lmdb)
        logger.info("Output dir: %s", output_dir)

    train_dataset = LmdbSampleDataset(str(args.train_lmdb), config=config, is_train=True)
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=config.DATA.NUM_WORKERS > 0,
        prefetch_factor=config.DATA.PREFETCH_FACTOR if config.DATA.NUM_WORKERS > 0 else None,
    )

    model = build_cls_model(config).cuda()
    criterion = build_loss(config).cuda()

    resume_path = args.resume
    if resume_path is None and not args.no_auto_resume:
        resume_path = auto_resume_path(output_dir)

    if resume_path is None and args.pretrained is not None and args.pretrained.exists():
        load_pretrained_weights(model.get_vision_transformer(), args.pretrained, logger)
    elif resume_path is None:
        model.unfreeze_backbone()
        if is_main_process():
            logger.info("No pretrained checkpoint found at %s, training from scratch with unfrozen backbone.", args.pretrained)

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    scheduler = build_scheduler(config, optimizer, len(train_loader))
    scaler = create_grad_scaler(args.precision)

    start_epoch = 0
    if resume_path is not None and Path(resume_path).exists():
        start_epoch = load_resume_checkpoint(Path(resume_path), model, optimizer, scheduler, scaler)
        if is_main_process():
            logger.info("Resumed from %s at epoch %d.", resume_path, start_epoch)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info("Trainable params: %d", total_params)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_stats = train_one_epoch(
            config=config,
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            precision=args.precision,
            logger=logger,
        )

        barrier()
        if is_main_process():
            checkpoint_path = save_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
            )
            logger.info(
                "Epoch %d finished | loss %.4f | avg_step_time %.3fs | saved %s",
                epoch,
                epoch_stats["loss"],
                epoch_stats["time"],
                checkpoint_path,
            )
        barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
