import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from util import Logger


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def init_distributed(opt):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.gpu = int(os.environ.get('LOCAL_RANK', 0))
        opt.distributed = True
    elif opt.distributed:
        opt.rank = 0
        opt.world_size = torch.cuda.device_count()
        opt.gpu = opt.local_rank
        os.environ['MASTER_PORT'] = str(opt.master_port)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['WORLD_SIZE'] = str(opt.world_size)
        os.environ['RANK'] = str(opt.rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=opt.world_size, rank=opt.rank)
        torch.cuda.set_device(opt.gpu)
        return
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.gpu = opt.gpu_ids[0] if opt.gpu_ids else 0
        opt.distributed = False
        return

    torch.cuda.set_device(opt.gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=opt.world_size, rank=opt.rank)
    dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    opt = TrainOptions().parse()
    seed_torch(100)

    init_distributed(opt)
    is_main = (not opt.distributed) or opt.rank == 0

    if is_main:
        Logger(os.path.join(opt.checkpoints_dir, opt.name, 'logging.log'))
        print('  '.join(list(sys.argv)))

    # Dataloaders
    train_loader, train_sampler = create_dataloader(opt, lmdb_path=opt.train_lmdb, is_train=True, distributed=opt.distributed)
    val_loader: Optional[torch.utils.data.DataLoader] = None
    if opt.val_lmdb:
        val_loader, _ = create_dataloader(opt, lmdb_path=opt.val_lmdb, is_train=False, distributed=opt.distributed)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train")) if is_main else None
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val")) if (is_main and val_loader) else None

    opt.iters_per_epoch = len(train_loader)
    opt.steps_per_epoch = len(train_loader)
    model = Trainer(opt)
    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    opt.device = device
    model.model.to(device)

    if opt.distributed:
        model.model = torch.nn.parallel.DistributedDataParallel(model.model, device_ids=[opt.gpu], output_device=opt.gpu)

    if is_main:
        print(f'cwd: {os.getcwd()}')
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Length of train loader: {len(train_loader)}")

    for epoch in range(opt.niter):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{opt.niter}", disable=not is_main, dynamic_ncols=True)

        for i, data in enumerate(train_loader):
            model.set_input(data)
            model.optimize_parameters()

            loss_val = model.loss.item() if hasattr(model.loss, 'item') else float(model.loss)
            progress.set_postfix(loss=f"{loss_val:.4f}", lr=f"{model.get_current_lr():.2e}")
            progress.update(1)

            if train_writer is not None:
                train_writer.add_scalar('loss', loss_val, model.total_steps)

        progress.close()

        if is_main:
            model.save_networks(epoch)

        if val_loader is not None:
            model.eval()
            from validate import validate

            metrics, _, _ = validate(model.model, opt, val_loader)
            if is_main:
                val_writer.add_scalar('accuracy', metrics['acc'], model.total_steps)
                val_writer.add_scalar('ap', metrics['pr_auc'], model.total_steps)
                val_writer.add_scalar('auroc', metrics['auroc'], model.total_steps)
            model.train()

    if is_main:
        model.eval()
        model.save_networks('last')

    cleanup_distributed()


if __name__ == '__main__':
    main()
