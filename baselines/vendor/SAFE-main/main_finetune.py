import os, sys, pdb
import csv
import random
import argparse
import datetime
import numpy as np
import time
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

import timm
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from models.resnet import resnet50
from data.datasets import TrainDataset
from engine_finetune import train_one_epoch, evaluate

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool, remap_checkpoint_keys
from test_logger import log_test_run


def get_args_parser():
    parser = argparse.ArgumentParser('Fine-tuning', add_help=False)
    parser.add_argument('--jpeg_factor', type=int, default=None)
    parser.add_argument('--blur_sigma', type=float, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--mask_patch_size', type=int, default=None)
    parser.add_argument('--transform_mode', type=str, default='crop')

    # Training Config
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='SAFE', type=str, metavar='MODEL',
                        help='model architecture')
    parser.add_argument('--input_size', default=256, type=int,
                        help='image input size')
    parser.add_argument('--layer_decay_type', type=str, choices=['single', 'group'], default='single',
                        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                        whereas the group strategy assigns the same decaying value for three consecutive layers""")

    # EMA related parameters
    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', action='store_true')
    parser.add_argument('--model_ema_eval', action='store_true', help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-2, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (defaults to min_lr_ratio of lr)')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--warmup_steps', type=int, default=2000, metavar='N',
                        help='num of steps to warmup LR (capped at one epoch)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01, metavar='LR',
                        help='final learning rate ratio for cosine annealing')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--pretrained', default=True, help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--head_init_scale', default=0.001, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--num_train', default=10000000000, type=int,
                        help="Number of training images, incluing real and fake")
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./checkpoints-5class',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=100, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--disable_eval', action='store_true',
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # GPU selection parameter
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id for distributed training')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true', help='Use ITP based distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--find_unused_parameters', action='store_true')

    parser.add_argument('--use_amp', action='store_true',
                        help="Use apex AMP (Automatic Mixed Precision) or not")

    return parser


def seed_everything(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    if args.gpu_ids == '-1' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        if args.distributed:
            torch.cuda.set_device(args.gpu)
            device = torch.device('cuda', args.gpu)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
            device = torch.device('cuda')
    cudnn.benchmark = True

    # Fix the Seed for Reproducibility
    if args.seed is not None:
        seed_everything(args.seed, True)

    # Init Train & Test Datasets
    if not args.eval:
        dataset_train = TrainDataset(is_train=True, args=args)
        if args.distributed:
            sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        if args.disable_eval or args.eval_data_path is None:
            dataset_val = None
            sampler_val = None
        else:
            dataset_val = TrainDataset(is_train=False, args=args)
            sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dataset_val)

        if args.log_dir is not None and utils.is_main_process():
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        if dataset_val is not None:
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=1 if args.transform_mode == 'source' else args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
        else:
            data_loader_val = None

    # Init Model
    if args.model == 'SAFE':
        model = resnet50(num_classes=2)
    else:
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=2)
    model.to(device)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        if utils.is_main_process():
            print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    if not args.eval:
        num_training_steps_per_epoch = len(data_loader_train)
        args.iters_per_epoch = num_training_steps_per_epoch
        args.total_training_steps = args.epochs * num_training_steps_per_epoch

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    # Enforce a cosine annealing floor.
    args.min_lr = args.lr * args.min_lr_ratio

    if utils.is_main_process():
        print(f"Number of params: {n_parameters / 1e6:.2f}M")
        print("Base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("Actual lr: %.2e" % args.lr)
        print("Accumulate grad iterations: %d" % args.update_freq)
        print("Effective batch size: %d" % eff_batch_size)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        assert args.layer_decay_type in ['single', 'group']
        if args.layer_decay_type == 'group':  # applies for Base and Large models
            num_layers = 12
        else:
            num_layers = sum(model_without_ddp.depths)
            print("--------------------------------------")
            print(num_layers)
            print("--------------------------------------")
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            depths=model_without_ddp.depths, layer_decay_type=args.layer_decay_type)
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        model.eval()
        if utils.is_main_process():
            print("Eval only mode")

        eval_roots = [p for p in str(args.eval_data_path).split(',') if p]
        rows = [[f"{args.resume} model testing..."], ['testset', 'accuracy', 'auroc', 'pr_auc', 'avg_recall', 'tpr@5%fpr']]

        for eval_root in eval_roots:
            args.eval_data_path = eval_root
            dataset_val = TrainDataset(is_train=False, args=args)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=1 if args.transform_mode == 'source' else args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            test_stats, metrics = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            if utils.is_main_process():
                print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.2%}")
                print(
                    f"test dataset: {eval_root} acc: {metrics['acc']:.2%}, auroc: {metrics['auroc']:.2%}, "
                    f"pr_auc: {metrics['pr_auc']:.2%}, avg_recall: {metrics['avg_recall']:.2%}, "
                    f"tpr@0.05fpr: {metrics['tpr_at_fpr']:.2%}"
                )
                print("***********************************")
                rows.append([
                    eval_root,
                    metrics['acc'] * 100,
                    metrics['auroc'] * 100,
                    metrics['pr_auc'] * 100,
                    metrics['avg_recall'] * 100,
                    metrics['tpr_at_fpr'] * 100,
                ])
                log_test_run(
                    project_root=Path(__file__).resolve().parent,
                    project_name=Path(__file__).resolve().parent.name,
                    dataset_path=eval_root,
                    ckpt_path=args.resume,
                    metrics=metrics,
                    cmd=" ".join(sys.argv),
                )

        if utils.is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
            csv_name = os.path.join(args.output_dir, f'{os.path.basename(args.resume)}_eval.csv')
            with open(csv_name, 'w') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerows(rows)
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    if utils.is_main_process():
        print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, args=args,
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch='last', model_ema=model_ema)
        if data_loader_val is not None:
            test_stats, metrics = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            if utils.is_main_process():
                print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.2%}")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            if utils.is_main_process():
                print(f'Max accuracy: {max_accuracy:.2%}')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
                log_writer.update(test_pr_auc=metrics['pr_auc'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         **{f'metric_{k}': v for k, v in metrics.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.model_ema and args.model_ema_eval:
                test_stats_ema, metrics_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                if utils.is_main_process():
                    print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.2%}")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    if utils.is_main_process():
                        print(f'Max EMA accuracy: {max_accuracy_ema:.2%}')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                    log_writer.update(test_pr_auc_ema=metrics_ema['pr_auc'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
                log_stats.update({**{f'metric_{k}_ema': v for k, v in metrics_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if utils.is_main_process():
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fine-tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
