import copy
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from validate import validate


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_distributed(opt):
    env_rank = os.environ.get("RANK")
    env_world = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")

    if env_rank is not None and env_world is not None:
        opt.rank = int(env_rank)
        opt.world_size = int(env_world)
        opt.local_rank = int(local_rank_env) if local_rank_env is not None else 0
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(opt.local_rank)
        opt.distributed = True
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.local_rank = 0
        opt.distributed = False

    opt.device = torch.device("cuda", opt.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    opt.gpu_ids = [opt.local_rank] if torch.cuda.is_available() else []
    return opt


def build_val_opt(train_opt):
    val_opt = copy.deepcopy(train_opt)
    val_opt.isTrain = False
    val_opt.serial_batches = True
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.data_aug = False
    return val_opt


if __name__ == "__main__":
    should_print_options = os.environ.get("RANK", "0") == "0"
    opt = TrainOptions().parse(print_options=should_print_options)
    opt.epochs = opt.epochs or opt.niter
    opt = init_distributed(opt)
    seed_torch(opt.seed + opt.rank)

    train_loader, train_sampler = create_dataloader(
        opt,
        lmdb_path=opt.train_lmdb,
        is_train=True,
        distributed=opt.distributed,
    )

    val_loader = None
    if opt.val_lmdb:
        val_opt = build_val_opt(opt)
        val_loader, _ = create_dataloader(
            val_opt,
            lmdb_path=opt.val_lmdb,
            is_train=False,
            distributed=opt.distributed,
        )

    if opt.rank == 0:
        print(" ".join(sys.argv))
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader is not None:
            print(f"Validation samples: {len(val_loader.dataset)}")

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train")) if opt.rank == 0 else None
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val")) if (opt.rank == 0 and val_loader is not None) else None

    model = Trainer(opt)
    if opt.distributed:
        model.model = torch.nn.parallel.DistributedDataParallel(
            model.model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=opt.find_unused_parameters,
        )

    start_epoch = opt.epoch_count
    end_epoch = start_epoch + opt.epochs - 1
    stop_training = False

    for epoch in range(start_epoch, end_epoch + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        model.train()
        progress = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch}/{end_epoch}",
            dynamic_ncols=True,
            disable=opt.rank != 0,
        )

        for batch in train_loader:
            model.set_input(batch)
            model.optimize_parameters()

            if opt.rank == 0:
                loss = float(model.loss.detach().item())
                loss1 = float(model.loss1.detach().item())
                loss2 = float(model.loss2.detach().item())
                lr = model.get_current_lr()

                if train_writer:
                    train_writer.add_scalar("loss/total", loss, model.total_steps)
                    train_writer.add_scalar("loss/contrastive", loss1, model.total_steps)
                    train_writer.add_scalar("loss/classification", loss2, model.total_steps)
                    train_writer.add_scalar("lr", lr, model.total_steps)

                progress.set_postfix(loss=f"{loss:.4f}", ctr=f"{loss1:.4f}", cla=f"{loss2:.4f}", lr=f"{lr:.2e}")
                progress.update(1)

            if opt.total_steps > 0 and model.total_steps >= opt.total_steps:
                stop_training = True
                break

        progress.close()

        if epoch % opt.delr_freq == 0 and epoch != start_epoch:
            if opt.rank == 0:
                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} changing lr at the end of epoch {epoch}"
                )
            model.adjust_learning_rate()

        if opt.rank == 0 and epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)
            model.save_networks("latest")

        if val_loader is not None:
            model.eval()
            metrics, _, _ = validate(
                model.model,
                val_loader,
                device=opt.device,
                distributed=opt.distributed,
                world_size=opt.world_size,
                rank=opt.rank,
                desc=f"Val @ epoch {epoch}",
                amp=opt.amp,
                amp_dtype=opt.amp_dtype,
            )
            if opt.rank == 0 and metrics is not None:
                for key, value in metrics.items():
                    if val_writer:
                        val_writer.add_scalar(key, value, model.total_steps)
                print(
                    f"(Val @ epoch {epoch}) "
                    + "; ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
                )
            model.train()

        if stop_training:
            break

    if opt.rank == 0:
        model.save_networks("last")

    if opt.distributed:
        dist.destroy_process_group()
