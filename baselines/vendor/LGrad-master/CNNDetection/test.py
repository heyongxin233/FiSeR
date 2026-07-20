import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import logging

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from data import create_dataloader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from test_logger import log_test_run

def seed_torch(seed=1029):
    import random
    import numpy as np

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
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.local_rank = int(os.environ.get('LOCAL_RANK', opt.local_rank))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opt.local_rank)
        opt.distributed = True
    elif opt.distributed:
        dist.init_process_group(backend='nccl')
        opt.rank = dist.get_rank()
        opt.world_size = dist.get_world_size()
        torch.cuda.set_device(opt.local_rank)
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.local_rank = 0
        opt.distributed = False

    opt.device = torch.device('cuda', opt.local_rank) if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    seed_torch(100)
    init_distributed(opt)

    logging.basicConfig(filename='log_test.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()

    if not opt.lmdb_path:
        raise ValueError('Please provide --lmdb_path pointing to one or more LMDB test sets.')

    lmdb_paths = [p.strip() for p in opt.lmdb_path.split(',') if p.strip()]
    names = [n.strip() for n in opt.eval_names.split(',')] if opt.eval_names else lmdb_paths

    if len(names) != len(lmdb_paths):
        raise ValueError('Length of eval_names must match lmdb_path when provided.')

    log_msg = f'Model_path {opt.model_path}'
    print(log_msg)
    logger.info(log_msg)

    state = torch.load(opt.model_path, map_location=opt.device)
    model = resnet50(num_classes=1)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.to(opt.device)

    if opt.distributed and opt.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
    model.eval()

    for name, path in zip(names, lmdb_paths):
        if opt.rank == 0:
            print('=' * 20)
            print(f"Evaluating {name} from {path}")
            logger.info(name)

        dataloader, _ = create_dataloader(opt, lmdb_path=path, is_train=False, distributed=opt.distributed)
        metrics, y_true, y_pred = validate(
            model,
            dataloader,
            device=opt.device,
            distributed=opt.distributed,
            world_size=opt.world_size,
            rank=opt.rank,
            desc=f"Testing {name}")

        if opt.rank == 0 and metrics is not None:
            log_msg = (f"({name}) acc: {metrics['acc'] * 100:.4f}; pr_auc: {metrics['pr_auc'] * 100:.4f}; "
                       f"auroc: {metrics['auroc'] * 100:.4f}; avg_recall: {metrics['avg_recall'] * 100:.4f}; "
                       f"tpr@0.05fpr: {metrics['tpr_at_fpr'] * 100:.4f}")
            print(log_msg)
            logger.info(log_msg)
            print('*' * 25)
            logger.info('*' * 25)
            log_test_run(
                project_root=Path(__file__).resolve().parents[1],
                project_name=Path(__file__).resolve().parents[1].name,
                dataset_path=path,
                ckpt_path=opt.model_path,
                metrics=metrics,
                cmd=" ".join(sys.argv),
            )

    if opt.distributed:
        dist.destroy_process_group()
