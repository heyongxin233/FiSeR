import os
import shutil
import datetime
import argparse
import numpy as np
import logging as logger
import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import importlib
from tqdm import tqdm
import torch.distributed as dist

from lmdb_utils import LMDBDataset
from metric_utils import evaluate_metrics
from test_logger import log_test_run


logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        args.distributed = True
    elif args.distributed:
        dist.init_process_group(backend='nccl')
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        torch.cuda.set_device(args.local_rank)
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    args.device = torch.device('cuda', args.local_rank) if torch.cuda.is_available() else torch.device('cpu')


def cosine_warmup_lr(step_idx, total_steps, base_lr, warmup_steps, min_lr_ratio):
    warmup_steps = min(warmup_steps, total_steps)
    if warmup_steps > 0 and step_idx < warmup_steps:
        return base_lr * float(step_idx + 1) / float(max(1, warmup_steps))
    progress = 0.0
    if total_steps > warmup_steps:
        progress = (step_idx - warmup_steps) / float(total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)


def train_one_epoch(data_loader, model, optimizer, cur_epoch, loss_meter, args):
    loss_meter.reset()
    model.train()
    progress_bar = tqdm(total=len(data_loader), desc=f"Train {cur_epoch}/{args.epoches}",
                        dynamic_ncols=True, disable=args.rank != 0)
    last_loss = None

    for step, (images, labels) in enumerate(data_loader):
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True).flatten().long()

        _, features_logits = model(images)

        loss_img = args.criterion_ce(features_logits, labels)
        labels_t = labels.t()
        text_feats = features_logits.t()
        tmp_loss = []
        for tmp_class_idx in range(args.num_class):
            cur_tmp_loss = [text_feats[tmp_class_idx][labels_t == tmp_class_idx].mean().unsqueeze(0)]
            for cur_tmp_inner_idx in range(args.num_class):
                if cur_tmp_inner_idx == tmp_class_idx:
                    continue
                cur_tmp_loss.append(text_feats[tmp_class_idx][labels_t == cur_tmp_inner_idx].mean().unsqueeze(0))
            tmp_loss.append(torch.cat(cur_tmp_loss))
        loss_text = args.criterion_ce(torch.stack(tmp_loss),
                                      torch.zeros(args.num_class).long().to(labels.device))

        loss = (loss_img + loss_text) / 2 if not torch.isnan(loss_text).any() else loss_img

        current_lr = cosine_warmup_lr(
            args.total_steps, args.total_train_steps, args.lr, args.warmup_steps, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group['lr'] = current_lr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        args.total_steps += 1

        loss_meter.update(loss.item(), images.shape[0])
        last_loss = loss.item()
        if progress_bar is not None:
            progress_bar.set_postfix(
                loss=f"{last_loss:.4f}",
                avg_loss=f"{loss_meter.avg:.4f}",
                lr=f"{current_lr:.6e}"
            )
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()
    return loss_meter.avg


def _gather_tensor(tensor, world_size):
    if world_size == 1:
        return tensor
    tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors, dim=0)


def _score_from_logits(logits, labels, num_class):
    probs = torch.softmax(logits, dim=1) if logits.dim() > 1 else torch.sigmoid(logits)
    if num_class == 1 or logits.dim() == 1:
        scores = probs.view(-1)
        binary_labels = labels.view(-1).float()
        return scores, binary_labels

    if num_class == 2:
        scores = probs[:, 1]
        binary_labels = labels.view(-1).float()
        return scores, binary_labels

    fake_indices = [idx for idx in range(num_class) if idx % 2 == 1]
    scores = probs[:, fake_indices].sum(dim=1)
    binary_labels = (labels.view(-1) % 2).float()
    return scores, binary_labels


def evaluate(model, data_loader, args, desc="Eval"):
    model.eval()
    all_scores = []
    all_labels = []
    progress_bar = tqdm(total=len(data_loader), desc=desc, dynamic_ncols=True, disable=args.rank != 0)
    for images, labels in data_loader:
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True).long()
        with torch.no_grad():
            _, logits = model(images)

        scores, binary_labels = _score_from_logits(logits, labels, args.num_class)
        all_scores.append(scores.detach())
        all_labels.append(binary_labels.detach())

        if progress_bar is not None:
            progress_bar.update(1)

    if all_scores:
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        if args.distributed:
            scores = _gather_tensor(scores, args.world_size)
            labels = _gather_tensor(labels, args.world_size)
        metrics = evaluate_metrics(labels.cpu().numpy(), scores.cpu().numpy())
    else:
        metrics = None

    if progress_bar is not None and metrics is not None:
        progress_bar.set_postfix(
            {
                "acc": f"{metrics['acc'] * 100:.2f}",
                "pr_auc": f"{metrics['pr_auc'] * 100:.2f}",
                "auroc": f"{metrics['auroc'] * 100:.2f}",
                "avg_recall": f"{metrics['avg_recall'] * 100:.2f}",
                "tpr@0.05": f"{metrics['tpr_at_fpr'] * 100:.2f}",
            }
        )
        progress_bar.refresh()
        progress_bar.close()
    elif progress_bar is not None:
        progress_bar.close()

    model.train()
    return metrics


def main(args):
    if args.isTrain == 1 and not args.train_lmdb:
        raise ValueError("Training requires --train_lmdb.")
    if args.isTrain == 0 and not args.test_lmdb:
        raise ValueError("Testing requires --test_lmdb.")

    train_loader = None
    train_sampler = None
    if args.isTrain == 1:
        train_dataset = LMDBDataset(args.train_lmdb, data_size=args.data_size, is_train=True)
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            num_workers=min(48, args.batch_size),
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True
        )

    val_loader = None
    if args.val_lmdb:
        val_dataset = LMDBDataset(args.val_lmdb, data_size=args.data_size, is_train=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(48, args.batch_size),
            sampler=val_sampler,
            drop_last=False,
            pin_memory=True
        )

    model = getattr(importlib.import_module('model'), args.model)(num_class=args.num_class)
    model = model.to(args.device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.rank == 0:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Params: %.2f' % (params / (1024 ** 2)))

    if args.resume:
        pretrained = torch.load(args.resume, map_location='cpu')
        if isinstance(pretrained, dict) and any(key.startswith('module.') for key in pretrained):
            pretrained = {key.replace('module.', '', 1): value for key, value in pretrained.items()}
        model.load_state_dict(pretrained)

    args.criterion_ce = torch.nn.CrossEntropyLoss().to(args.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.lr)

    loss_meter = AverageMeter()
    args.total_steps = 0
    if train_loader is not None:
        args.total_train_steps = len(train_loader) * args.epoches

    for epoch in range(1, args.epoches + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if args.isTrain == 1:
            train_one_epoch(train_loader, model, optimizer, epoch, loss_meter, args)

        if val_loader is not None:
            evaluate(model, val_loader, args, desc=f"Val {epoch}/{args.epoches}")

        if args.isTrain == 1 and args.rank == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_epoch_{epoch}.pth"))

        if args.isTrain == 0:
            break

    if args.test_lmdb:
        test_paths = [p.strip() for p in args.test_lmdb.split(',') if p.strip()]
        names = [n.strip() for n in args.eval_names.split(',')] if args.eval_names else test_paths
        if len(names) != len(test_paths):
            raise ValueError("Length of eval_names must match test_lmdb when provided.")
        for name, path in zip(names, test_paths):
            test_dataset = LMDBDataset(path, data_size=args.data_size, is_train=False)
            test_sampler = DistributedSampler(test_dataset, shuffle=False) if args.distributed else None
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=min(48, args.batch_size),
                sampler=test_sampler,
                drop_last=False,
                pin_memory=True
            )
            metrics = evaluate(model, test_loader, args, desc=f"Test {name}")
            if args.rank == 0 and metrics is not None:
                print(
                    f"Test {name} metrics -> "
                    f"acc: {metrics['acc'] * 100:.2f}%, "
                    f"pr_auc: {metrics['pr_auc'] * 100:.2f}%, "
                    f"auroc: {metrics['auroc'] * 100:.2f}%, "
                    f"avg_recall: {metrics['avg_recall'] * 100:.2f}%, "
                    f"tpr@0.05fpr: {metrics['tpr_at_fpr'] * 100:.2f}%"
                )
                log_test_run(
                    project_root=Path(__file__).resolve().parent,
                    project_name=Path(__file__).resolve().parent.name,
                    dataset_path=path,
                    ckpt_path=args.resume if args.resume else None,
                    metrics=metrics,
                    cmd=" ".join(sys.argv),
                )


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    conf = argparse.ArgumentParser()
    conf.add_argument('--train_lmdb', type=str, default=None, help='Path to training LMDB.')
    conf.add_argument('--val_lmdb', type=str, default=None, help='Optional path to validation LMDB.')
    conf.add_argument('--test_lmdb', type=str, default=None, help='Comma-separated LMDBs for testing.')
    conf.add_argument('--eval_names', type=str, default=None, help='Optional comma-separated names for test_lmdb.')
    conf.add_argument('--isTrain', type=int, default=1, help='1 for train, 0 for test only.')
    conf.add_argument("--model", type=str, default='LASTED')
    conf.add_argument("--num_class", type=int, default=2, help='The class number of training dataset')
    conf.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    conf.add_argument('--min_lr_ratio', type=float, default=0.01, help='Final LR ratio for cosine decay.')
    conf.add_argument('--warmup_steps', type=int, default=2000, help='Number of warmup steps (capped to one epoch).')
    conf.add_argument("--weights", type=str, default='out_dir', help="The folder to save models.")
    conf.add_argument('--epoches', type=int, default=30, help='The training epoches.')
    conf.add_argument('--batch_size', type=int, default=48, help='The training batch size over all gpus.')
    conf.add_argument('--data_size', type=int, default=448, help='The image size for training.')
    conf.add_argument('--gpu', type=str, default=None, help='Optional CUDA_VISIBLE_DEVICES override.')
    conf.add_argument("--resume", type=str, default='')
    conf.add_argument('--distributed', action='store_true', help='Enable DDP training.')
    conf.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training.')
    args = conf.parse_args()
    os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), os.cpu_count()))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    init_distributed(args)
    args.out_dir = args.weights

    if args.isTrain == 1:
        date_now = datetime.datetime.now()
        date_now = '/Log_v%02d%02d%02d%02d' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
        args.time = date_now
        args.out_dir = args.out_dir + args.time
        if args.rank == 0:
            if os.path.exists(args.out_dir):
                shutil.rmtree(args.out_dir)
            os.makedirs(args.out_dir, exist_ok=True)

    if args.rank == 0:
        logger.info(args)
    main(args)

    if args.distributed:
        dist.destroy_process_group()
