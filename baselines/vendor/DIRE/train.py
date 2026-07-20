from utils.config import cfg  # isort: split

import os
import sys
import time

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.earlystop import EarlyStopping
from utils.eval import get_val_cfg, validate
from utils.trainer import Trainer
from utils.utils import Logger


def init_distributed(cfg):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", cfg.local_rank))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(cfg.local_rank)
        cfg.distributed = True
    elif cfg.distributed:
        dist.init_process_group(backend="nccl")
        cfg.rank = dist.get_rank()
        cfg.world_size = dist.get_world_size()
        torch.cuda.set_device(cfg.local_rank)
    else:
        cfg.rank = 0
        cfg.world_size = 1
        cfg.local_rank = 0
        cfg.distributed = False

    cfg.device = torch.device("cuda", cfg.local_rank) if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    init_distributed(cfg)
    if cfg.train_lmdb is None:
        cfg.dataset_root = os.path.join(cfg.dataset_root, "train")

    data_loader = create_dataloader(cfg, lmdb_path=cfg.train_lmdb, is_train=True, distributed=cfg.distributed)
    dataset_size = len(data_loader)

    log = Logger()
    def log_message(message: str):
        if cfg.rank == 0:
            log.write(message, is_terminal=0)
            tqdm.write(message.rstrip("\n"))

    if cfg.rank == 0:
        log.open(cfg.logs_path, mode="a")
        log_message("Num of training images = %d\n" % (dataset_size * cfg.batch_size))
        log_message("Config:\n" + str(cfg.to_dict()) + "\n")

    train_writer = SummaryWriter(os.path.join(cfg.exp_dir, "train")) if cfg.rank == 0 else None
    val_writer = SummaryWriter(os.path.join(cfg.exp_dir, "val")) if cfg.rank == 0 else None

    val_cfg = None
    val_loader = None
    if cfg.val_lmdb:
        val_cfg = get_val_cfg(cfg, split="val", copy=True)
        val_loader = create_dataloader(val_cfg, lmdb_path=cfg.val_lmdb, is_train=False, distributed=cfg.distributed)

    trainer = Trainer(cfg)
    if cfg.distributed:
        trainer.model = torch.nn.parallel.DistributedDataParallel(
            trainer.model, device_ids=[cfg.local_rank], output_device=cfg.local_rank
        )
    trainer.train()
    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.001, verbose=True) if cfg.earlystop else None

    if cfg.rank == 0:
        log_message("  ".join(list(sys.argv)) + "\n")
        log_message(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Length of data loader: {len(data_loader)}\n"
        )

    for epoch in range(cfg.epoch_count, cfg.nepoch + 1):
        if cfg.distributed and hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)

        progress_bar = (
            tqdm(total=len(data_loader), desc=f"Epoch {epoch}/{cfg.nepoch}", dynamic_ncols=True)
            if cfg.rank == 0
            else None
        )
        running_loss = 0.0
        last_loss = None

        for step, data in enumerate(data_loader, start=1):
            trainer.total_steps += 1
            trainer.set_input(data)
            trainer.optimize_parameters()

            loss_val = trainer.loss.item() if torch.is_tensor(trainer.loss) else float(trainer.loss)
            running_loss += loss_val
            last_loss = loss_val
            avg_loss = running_loss / step
            lr = trainer.optimizer.param_groups[0]["lr"]

            if train_writer and trainer.total_steps % cfg.loss_freq == 0:
                train_writer.add_scalar("loss", loss_val, trainer.total_steps)
                train_writer.add_scalar("lr", lr, trainer.total_steps)

            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{last_loss:.4f}" if last_loss is not None else "n/a",
                    avg_loss=f"{avg_loss:.4f}",
                    lr=f"{lr:.3e}",
                    step=trainer.total_steps,
                )
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        if cfg.rank == 0:
            log_message("saving the model at the end of epoch %d, iters %d\n" % (epoch, trainer.total_steps))
            trainer.save_networks("latest")
            trainer.save_networks(epoch)

        if val_loader is not None:
            trainer.eval()
            metrics, _, _ = validate(
                trainer.model,
                val_loader,
                device=cfg.device,
                distributed=cfg.distributed,
                world_size=cfg.world_size,
                rank=cfg.rank,
                desc=f"Val {epoch}/{cfg.nepoch}",
            )
            if cfg.rank == 0 and metrics is not None:
                val_writer.add_scalar("acc", metrics["acc"], trainer.total_steps)
                val_writer.add_scalar("pr_auc", metrics["pr_auc"], trainer.total_steps)
                val_writer.add_scalar("auroc", metrics["auroc"], trainer.total_steps)
                log_message(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} (Val @ epoch {epoch}) "
                    f"acc: {metrics['acc']}; pr_auc: {metrics['pr_auc']}; auroc: {metrics['auroc']}; "
                    f"avg_recall: {metrics['avg_recall']}\n"
                )

            if early_stopping and metrics is not None:
                early_stopping(metrics["acc"], trainer)
                if early_stopping.early_stop:
                    if trainer.adjust_learning_rate():
                        log_message("Learning rate dropped by 10, continue training...\n")
                        early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.002, verbose=True)
                    else:
                        log_message("Early stopping.\n")
                        break
            trainer.train()

        if trainer.scheduler is not None:
            trainer.scheduler.step()

    if cfg.rank == 0:
        trainer.save_networks("last")

    if cfg.distributed:
        dist.destroy_process_group()
