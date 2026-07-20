import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import ImageFile

from config import ConfigurationManager as Configurator
from loader import create_dataloader
from model import model as NeuralNetwork
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


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint, strict=True)


def main():
    manager = Configurator()
    config = manager.parse(display_settings=False)
    config.isTrain = False
    config.isVal = True
    config = init_distributed(config)
    if config.rank == 0:
        manager.display_configuration(config)
    seed_everything(config.seed + config.rank)

    checkpoint_path = config.model_path or config.load
    if not checkpoint_path:
        raise ValueError('Please provide --model_path for evaluation.')
    if not config.lmdb_path:
        raise ValueError('Please provide --lmdb_path for evaluation.')

    lmdb_paths = [path.strip() for path in config.lmdb_path.split(',') if path.strip()]
    eval_names = [name.strip() for name in config.eval_names.split(',') if name.strip()]
    if eval_names and len(eval_names) != len(lmdb_paths):
        raise ValueError('Length of --eval_names must match --lmdb_path when provided.')
    if not eval_names:
        eval_names = [os.path.basename(path.rstrip('/')) for path in lmdb_paths]

    model = NeuralNetwork(pretrain=False, pretrained_dir=config.pretrained_dir).to(config.device)
    load_checkpoint(model, checkpoint_path, config.device)
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
        )

    if config.rank == 0:
        print(' '.join(sys.argv))
        print(f'Loaded checkpoint: {checkpoint_path}')

    summaries = []
    for dataset_index, (dataset_name, lmdb_path) in enumerate(zip(eval_names, lmdb_paths), start=1):
        data_loader, sampler = create_dataloader(
            config,
            lmdb_path=lmdb_path,
            is_train=False,
            distributed=config.distributed,
        )
        if sampler is not None:
            sampler.set_epoch(dataset_index)

        if config.rank == 0:
            print(f'[{dataset_name}] samples: {len(data_loader.dataset)}')

        metrics, _, _ = validate(
            model,
            data_loader,
            device=config.device,
            distributed=config.distributed,
            world_size=config.world_size,
            rank=config.rank,
            desc=f'Test {dataset_name}',
            precision=config.precision,
            max_batches=config.max_eval_batches,
        )

        if config.rank == 0:
            print(f'[{dataset_name}]')
            for key, value in metrics.items():
                print(f'  {key}: {value:.6f}')
            summaries.append(
                {
                    'dataset': dataset_name,
                    'lmdb_path': lmdb_path,
                    'checkpoint': checkpoint_path,
                    'metrics': metrics,
                }
            )

    if config.rank == 0:
        os.makedirs(config.results_dir, exist_ok=True)
        results_path = Path(config.results_dir) / f'{Path(checkpoint_path).stem}_metrics.json'
        with open(results_path, 'w', encoding='utf-8') as file_obj:
            json.dump(summaries, file_obj, indent=2, ensure_ascii=False)
        print(f'Saved metrics to {results_path}')

    if config.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
