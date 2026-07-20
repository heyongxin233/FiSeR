import os
import argparse
import time
import sys
from pathlib import Path
import torch
from tqdm import tqdm

from data import create_dataloader
from models import get_model
from validate import validate
from test_logger import log_test_run


def load_model(arch, model_path, device):
    model = get_model(arch)
    state = torch.load(model_path, map_location=device)
    state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--lmdb_path', type=str, required=True, help='comma separated lmdb paths for evaluation')
    parser.add_argument('--eval_names', type=str, default=None, help='optional comma separated names matching lmdb_path')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--distributed', action='store_true', help='enable distributed evaluation')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed evaluation')
    opt = parser.parse_args()

    if not opt.lmdb_path:
        raise ValueError('Please provide --lmdb_path pointing to one or more LMDB test sets.')

    opt.isTrain = False
    opt.serial_batches = True
    opt.class_bal = False
    opt.data_aug = False
    opt.no_flip = True
    opt.no_crop = False
    opt.no_resize = False
    opt.loadSize = 256
    opt.cropSize = 224
    opt.rz_interp = ['bilinear']
    opt.blur_prob = 0.5
    opt.blur_sig = [0.0, 3.0]
    opt.jpg_prob = 0.5
    opt.jpg_method = ['cv2', 'pil']
    opt.jpg_qual = list(range(30, 101))

    lmdb_paths = [p.strip() for p in opt.lmdb_path.split(',') if p.strip()]
    names = [n.strip() for n in opt.eval_names.split(',')] if opt.eval_names else lmdb_paths
    if len(names) != len(lmdb_paths):
        raise ValueError('Length of eval_names must match lmdb_path when provided.')

    if opt.distributed:
        torch.distributed.init_process_group(backend='nccl')
        opt.rank = torch.distributed.get_rank()
        opt.world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(opt.local_rank)
    else:
        opt.rank = 0
        opt.world_size = 1

    device = torch.device('cuda', opt.local_rank) if torch.cuda.is_available() else torch.device('cpu')

    model = load_model(opt.arch, opt.model_path, device)

    if opt.rank == 0:
        tqdm.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Loaded model {opt.model_path}")

    for name, path in zip(names, lmdb_paths):
        dataloader, _ = create_dataloader(opt, lmdb_path=path, is_train=False, distributed=opt.distributed)
        metrics, _, _ = validate(
            model,
            dataloader,
            device=device,
            distributed=opt.distributed,
            world_size=opt.world_size,
            rank=opt.rank,
            desc=f"Test {name}")
        if opt.rank == 0 and metrics is not None:
            log_msg = (f"({name}) acc: {metrics['acc'] * 100:.2f}; pr_auc: {metrics['pr_auc'] * 100:.2f}; "
                       f"auroc: {metrics['auroc'] * 100:.2f}; avg_recall: {metrics['avg_recall'] * 100:.2f}; "
                       f"tpr@0.05fpr: {metrics['tpr_at_fpr'] * 100:.2f}")
            tqdm.write(log_msg)
            log_test_run(
                project_root=Path(__file__).resolve().parent,
                project_name=Path(__file__).resolve().parent.name,
                dataset_path=path,
                ckpt_path=opt.model_path,
                metrics=metrics,
                cmd=" ".join(sys.argv),
            )

    if opt.distributed:
        torch.distributed.destroy_process_group()
