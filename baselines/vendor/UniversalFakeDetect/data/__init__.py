import os
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .datasets import RealFakeDataset, LMDBDataset

    

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def _select_data_path(opt, is_train, lmdb_path):
    if lmdb_path:
        return lmdb_path
    if is_train:
        return getattr(opt, 'train_lmdb', None)
    return getattr(opt, 'val_lmdb', None) or getattr(opt, 'lmdb_path', None)


def get_dataset(opt, is_train=True, lmdb_path=None):
    target_path = _select_data_path(opt, is_train, lmdb_path)
    if target_path:
        is_lmdb = target_path.endswith('.lmdb') or os.path.isfile(target_path) or os.path.isfile(os.path.join(target_path, 'data.mdb'))
        if is_lmdb:
            return LMDBDataset(opt, target_path, is_train=is_train)
    return RealFakeDataset(opt)


def create_dataloader(opt, preprocess=None, lmdb_path=None, is_train=None, distributed=None):
    is_train = opt.isTrain if is_train is None else is_train
    distributed = getattr(opt, 'distributed', False) if distributed is None else distributed

    shuffle = not opt.serial_batches if (is_train and not opt.class_bal) else False
    dataset = get_dataset(opt, is_train=is_train, lmdb_path=lmdb_path)
    if '2b' in opt.arch and hasattr(dataset, 'transform'):
        dataset.transform = preprocess

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, num_replicas=opt.world_size, rank=opt.rank, drop_last=False)
        shuffle = False
    elif opt.class_bal:
        sampler = get_bal_sampler(dataset)
        shuffle = False if sampler is not None else shuffle

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads),
                                              pin_memory=True)
    return data_loader, sampler
