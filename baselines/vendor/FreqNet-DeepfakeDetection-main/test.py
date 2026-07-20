import os
import random
import sys
from pathlib import Path

import torch
import numpy as np
import torch.distributed as dist

from data import create_dataloader
from networks.freqnet import freqnet
from options.test_options import TestOptions
from validate import validate
from test_logger import log_test_run


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
    opt = TestOptions().parse(print_options=False)
    init_distributed(opt)
    seed_torch(100)

    if not opt.lmdb_path:
        raise ValueError('Please provide --lmdb_path for evaluation')

    if opt.rank == 0:
        print('  '.join(list(sys.argv)))

    model = freqnet(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']

    # Handle checkpoints saved from DataParallel/DistributedDataParallel by
    # stripping the leading "module." prefix to match the single-GPU model
    # definition expected during evaluation.
    if any(key.startswith('module.') for key in state_dict):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(opt.device)
    model.eval()

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    data_loader, _ = create_dataloader(opt, lmdb_path=opt.lmdb_path, is_train=False, distributed=opt.distributed)

    metrics = validate(model, data_loader, device=opt.device, distributed=opt.distributed, world_size=opt.world_size)

    if opt.rank == 0:
        metric_str = '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print('Evaluation results:')
        print(metric_str)
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
