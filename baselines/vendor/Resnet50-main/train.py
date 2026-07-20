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
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def init_distributed(opt):
    env_rank = os.environ.get('RANK')
    env_world = os.environ.get('WORLD_SIZE')
    local_rank_env = os.environ.get('LOCAL_RANK')

    if env_rank is not None and env_world is not None:
        opt.rank = int(env_rank)
        opt.world_size = int(env_world)
        opt.local_rank = int(local_rank_env) if local_rank_env is not None else 0
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opt.local_rank)
        opt.distributed = True
    elif opt.distributed:
        os.environ.setdefault('MASTER_ADDR', opt.master_addr)
        os.environ.setdefault('MASTER_PORT', opt.master_port)
        opt.rank = int(env_rank) if env_rank is not None else opt.rank
        opt.world_size = int(env_world) if env_world is not None else opt.world_size
        opt.local_rank = int(local_rank_env) if local_rank_env is not None else opt.local_rank
        dist.init_process_group(backend='nccl', init_method='env://', world_size=opt.world_size, rank=opt.rank)
        torch.cuda.set_device(opt.local_rank)
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.local_rank = 0
        opt.distributed = False

    opt.device = torch.device('cuda', opt.local_rank) if torch.cuda.is_available() else torch.device('cpu')
    return opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt = init_distributed(opt)
    seed_torch(100 + opt.rank)

    train_loader, train_sampler = create_dataloader(opt, lmdb_path=opt.train_lmdb, is_train=True, distributed=opt.distributed)
    val_loader = None
    if opt.val_lmdb:
        val_loader, _ = create_dataloader(opt, lmdb_path=opt.val_lmdb, is_train=False, distributed=opt.distributed)

    if opt.rank == 0:
        print('  '.join(list(sys.argv)))
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader is not None:
            print(f"Validation samples: {len(val_loader.dataset)}")

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train")) if opt.rank == 0 else None
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val")) if (opt.rank == 0 and val_loader is not None) else None

    opt.iters_per_epoch = len(train_loader)
    opt.steps_per_epoch = len(train_loader)
    model = Trainer(opt)
    if opt.distributed:
        model.model = torch.nn.parallel.DistributedDataParallel(model.model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    for epoch in range(opt.epoch_count, opt.epoch_count + opt.niter):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{opt.epoch_count + opt.niter - 1}", dynamic_ncols=True, disable=opt.rank != 0)
        for _, data in enumerate(train_loader):
            model.set_input(data)
            model.optimize_parameters()

            current_loss = model.loss.item() if torch.is_tensor(model.loss) else float(model.loss)
            if train_writer:
                train_writer.add_scalar('loss', current_loss, model.total_steps)
            progress.set_postfix(loss=f"{current_loss:.4f}", lr=f"{model.get_current_lr():.6f}")
            progress.update(1)
        progress.close()

        if opt.rank == 0:
            model.save_networks(epoch)
            model.save_networks('latest')

        if val_loader is not None:
            model.eval()
            metrics, _, _ = validate(model.model, val_loader, device=opt.device, distributed=opt.distributed, world_size=opt.world_size, rank=opt.rank, desc=f"Val @ epoch {epoch}")
            if opt.rank == 0:
                for k, v in metrics.items():
                    val_writer.add_scalar(k, v, model.total_steps)
                tqdm.write(f"(Val @ epoch {epoch}) " + '; '.join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            model.train()

    model.eval()
    if opt.rank == 0:
        model.save_networks('last')

    if opt.distributed:
        dist.destroy_process_group()
