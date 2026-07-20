import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

from data import create_dataloader
from networks.resnet import resnet50
from options.test_options import TestOptions
from validate import validate
from test_logger import log_test_run


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
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
    seed_torch(100)
    opt = TestOptions().parse(print_options=False)
    opt = init_distributed(opt)

    dataloader, sampler = create_dataloader(opt, lmdb_path=opt.lmdb_path, is_train=False, distributed=opt.distributed)
    if sampler is not None:
        sampler.set_epoch(0)

    if opt.rank == 0:
        print('  '.join(list(sys.argv)))
        print(f"Testing samples: {len(dataloader.dataset)}")

    model = resnet50(num_classes=1)
    checkpoint = torch.load(opt.model_path, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(opt.device)
    model.eval()

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    metrics, y_true, y_pred = validate(model, dataloader, device=opt.device, distributed=opt.distributed, world_size=opt.world_size, rank=opt.rank, desc="Testing")

    if opt.rank == 0:
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        log_test_run(
            project_root=Path(__file__).resolve().parent,
            project_name=Path(__file__).resolve().parent.name,
            dataset_path=opt.lmdb_path,
            ckpt_path=opt.model_path,
            metrics=metrics,
            cmd=" ".join(sys.argv),
        )

    if opt.distributed:
        dist.destroy_process_group()
