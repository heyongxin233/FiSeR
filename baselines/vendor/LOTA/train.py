import os
import random
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from PIL import ImageFile
from tqdm import tqdm

import util as toolkit
from config import ConfigurationManager as Configurator
from loader import create_dataloader
from model import model as NeuralNetwork
from util import bceLoss as compute_binary_loss
from validate import validate

ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def init_distributed(config):
    env_rank = os.environ.get('RANK')
    env_world = os.environ.get('WORLD_SIZE')
    env_local_rank = os.environ.get('LOCAL_RANK')

    if env_rank is not None and env_world is not None:
        config.rank = int(env_rank)
        config.world_size = int(env_world)
        config.local_rank = int(env_local_rank) if env_local_rank is not None else 0
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(config.local_rank)
        config.distributed = True
    elif config.distributed:
        os.environ.setdefault('MASTER_ADDR', config.master_addr)
        os.environ.setdefault('MASTER_PORT', config.master_port)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=config.world_size,
            rank=config.rank,
        )
        torch.cuda.set_device(config.local_rank)
    else:
        config.rank = 0
        config.world_size = 1
        config.local_rank = 0
        config.distributed = False

    if torch.cuda.is_available():
        config.device = torch.device('cuda', config.local_rank)
    else:
        config.device = torch.device('cpu')
    return config


def get_amp_context(precision, device):
    if device.type != 'cuda' or precision == 'fp32':
        return nullcontext()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)


def get_state_dict(model):
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint, strict=True)


def main_execution():
    manager = Configurator()
    config = manager.parse(display_settings=False)
    config.isTrain = True
    config.isVal = False
    config = init_distributed(config)
    if config.rank == 0:
        manager.display_configuration(config)
    seed_everything(config.seed + config.rank)

    if not config.train_lmdb:
        raise ValueError('Please provide --train_lmdb for training.')

    os.makedirs(config.output_dir, exist_ok=True)

    train_loader, train_sampler = create_dataloader(
        config,
        lmdb_path=config.train_lmdb,
        is_train=True,
        distributed=config.distributed,
    )
    val_loader = None
    val_sampler = None
    if config.val_lmdb:
        val_loader, val_sampler = create_dataloader(
            config,
            lmdb_path=config.val_lmdb,
            is_train=False,
            distributed=config.distributed,
        )

    if config.rank == 0:
        print(' '.join(sys.argv))
        print(f'Training samples: {len(train_loader.dataset)}')
        if val_loader is not None:
            print(f'Validation samples: {len(val_loader.dataset)}')

    model = NeuralNetwork(
        pretrain=(not config.no_pretrained) and not config.load,
        pretrained_dir=config.pretrained_dir,
    ).to(config.device)
    if config.load:
        load_checkpoint(model, config.load, config.device)
        if config.rank == 0:
            print(f'Loaded checkpoint from {config.load}')

    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = compute_binary_loss()
    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=(config.device.type == 'cuda' and config.precision == 'fp16'),
    )

    stop_training = False
    global_step = 0
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(config.epochs * steps_per_epoch, 1)
    warmup_steps = max(config.warmup_epochs, 0) * steps_per_epoch

    for epoch in range(1, config.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if config.scheduler == 'poly':
            current_lr = toolkit.poly_lr(optimizer, config.lr, epoch - 1, max(config.epochs, 1))
        else:
            current_lr = optimizer.param_groups[0]['lr']
        model.train()
        progress = tqdm(
            total=len(train_loader),
            desc=f'Epoch {epoch}/{config.epochs}',
            dynamic_ncols=True,
            disable=config.rank != 0,
        )

        running_loss = 0.0
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(config.device, non_blocking=True)
            targets = targets.to(config.device, non_blocking=True).view(-1)

            if config.scheduler == 'cosine':
                current_lr = toolkit.cosine_warmup_lr(
                    optimizer,
                    config.lr,
                    config.min_lr,
                    global_step,
                    total_steps,
                    warmup_steps,
                )

            optimizer.zero_grad(set_to_none=True)
            with get_amp_context(config.precision, config.device):
                logits = model(images).view(-1)
                loss = criterion(logits, targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_value = float(loss.detach().item())
            running_loss += loss_value
            global_step += 1

            progress.set_postfix(loss=f'{loss_value:.4f}', lr=f'{current_lr:.2e}')
            progress.update(1)

            if config.max_steps > 0 and global_step >= config.max_steps:
                stop_training = True
                break

        progress.close()

        if config.rank == 0:
            epoch_loss = running_loss / max(batch_index, 1)
            tqdm.write(f'Epoch {epoch} finished | loss={epoch_loss:.6f} | lr={current_lr:.2e}')
            torch.save(get_state_dict(model), os.path.join(config.output_dir, f'model_epoch_{epoch}.pth'))
            torch.save(get_state_dict(model), os.path.join(config.output_dir, 'model_latest.pth'))

        if val_loader is not None:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            metrics, _, _ = validate(
                model,
                val_loader,
                device=config.device,
                distributed=config.distributed,
                world_size=config.world_size,
                rank=config.rank,
                desc=f'Val {epoch}/{config.epochs}',
                precision=config.precision,
                max_batches=config.max_eval_batches,
            )
            if config.rank == 0:
                tqdm.write(
                    f"Val {epoch} | "
                    + '; '.join(f'{key}={value:.6f}' for key, value in metrics.items())
                )

        if stop_training:
            break

    if config.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main_execution()
