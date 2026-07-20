import os

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import LMDBDataset, dataset_folder


def _select_data_path(opt, is_train, lmdb_path):
    if lmdb_path:
        return lmdb_path
    if is_train:
        return getattr(opt, "train_lmdb", None) or opt.dataroot
    return getattr(opt, "val_lmdb", None) or getattr(opt, "lmdb_path", None) or opt.dataroot


def get_dataset(opt, is_train=True, lmdb_path=None):
    target_path = _select_data_path(opt, is_train, lmdb_path)
    if target_path is None:
        raise ValueError("A valid lmdb or folder path must be provided.")

    is_lmdb = (
        target_path.endswith(".lmdb")
        or os.path.isfile(target_path)
        or os.path.isfile(os.path.join(target_path, "data.mdb"))
    )
    if is_lmdb:
        return LMDBDataset(opt, target_path, is_train=is_train)

    classes = os.listdir(target_path) if len(opt.classes) == 0 else opt.classes
    if "0_real" not in classes or "1_fake" not in classes:
        datasets = []
        for cls in classes:
            root = os.path.join(target_path, cls)
            datasets.append(dataset_folder(opt, root))
        return torch.utils.data.ConcatDataset(datasets)
    return dataset_folder(opt, target_path)


def get_bal_sampler(dataset):
    targets = []
    if hasattr(dataset, "datasets"):
        for sub_dataset in dataset.datasets:
            targets.extend(sub_dataset.targets)
    elif hasattr(dataset, "targets"):
        targets.extend(dataset.targets)
    else:
        return None

    ratio = np.bincount(targets)
    weights = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = weights[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataloader(opt, lmdb_path=None, is_train=None, distributed=None):
    is_train = opt.isTrain if is_train is None else is_train
    distributed = getattr(opt, "distributed", False) if distributed is None else distributed

    dataset = get_dataset(opt, is_train=is_train, lmdb_path=lmdb_path)
    sampler = None
    shuffle = False
    is_distributed = distributed and torch.distributed.is_available() and torch.distributed.is_initialized()

    if is_train and opt.class_bal:
        sampler = get_bal_sampler(dataset)
    elif is_distributed:
        shuffle = is_train and not opt.serial_batches
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            num_replicas=opt.world_size,
            rank=opt.rank,
            drop_last=is_train,
        )
    else:
        shuffle = is_train and not opt.serial_batches

    if sampler is not None:
        shuffle = False

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=is_train,
        num_workers=int(opt.num_threads),
        pin_memory=torch.cuda.is_available(),
    )
    return data_loader, sampler
