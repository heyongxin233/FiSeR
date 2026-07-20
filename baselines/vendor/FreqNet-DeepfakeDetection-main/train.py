import os
import random
import sys

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
        opt.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opt.local_rank)
        opt.distributed = True
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.local_rank = 0
        opt.distributed = False
    opt.device = torch.device('cuda', opt.local_rank) if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    opt = TrainOptions().parse()
    init_distributed(opt)
    seed_torch(100)

    if opt.rank == 0:
        os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
        print('  '.join(list(sys.argv)))

    train_loader, train_sampler = create_dataloader(opt, lmdb_path=opt.train_lmdb, is_train=True, distributed=opt.distributed)
    val_loader = None
    if opt.val_lmdb:
        val_loader, _ = create_dataloader(opt, lmdb_path=opt.val_lmdb, is_train=False, distributed=opt.distributed)

    if opt.rank == 0:
        train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
        val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val")) if val_loader else None
        print(f"Length of train loader: {len(train_loader)}")
    else:
        train_writer = None
        val_writer = None

    opt.iters_per_epoch = len(train_loader)
    opt.steps_per_epoch = len(train_loader)
    model = Trainer(opt)
    model.train()
    if opt.distributed:
        model.model = torch.nn.parallel.DistributedDataParallel(
            model.model, device_ids=[opt.local_rank], output_device=opt.local_rank
        )

    for epoch in range(opt.niter):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch}/{opt.niter - 1}",
            dynamic_ncols=True,
        ) if opt.rank == 0 else None
        for data in train_loader:
            model.set_input(data)
            model.optimize_parameters()

            current_loss = model.loss.item() if torch.is_tensor(model.loss) else float(model.loss)
            current_lr = model.current_learning_rate()
            if progress_bar is not None:
                progress_bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.6f}" if current_lr else None)

            if train_writer and model.total_steps % opt.loss_freq == 0:
                train_writer.add_scalar('loss', current_loss, model.total_steps)
                if current_lr is not None:
                    train_writer.add_scalar('lr', current_lr, model.total_steps)

            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        if opt.rank == 0 and ((epoch + 1) % opt.save_epoch_freq == 0):
            model.save_networks(epoch)

        model.eval()
        metrics = validate(
            model.model,
            val_loader,
            device=opt.device,
            distributed=opt.distributed,
            world_size=opt.world_size,
        ) if val_loader else None
        model.train()

        if metrics and opt.rank == 0:
            for key, value in metrics.items():
                if val_writer:
                    val_writer.add_scalar(key, value, model.total_steps)
            metric_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"(Val @ epoch {epoch}) {metric_str}")

    if opt.rank == 0:
        model.eval()
        model.save_networks('last')

    if opt.distributed:
        dist.destroy_process_group()
