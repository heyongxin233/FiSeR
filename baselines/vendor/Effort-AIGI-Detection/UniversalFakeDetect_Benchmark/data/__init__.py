import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import LMDBDataset, RealFakeDataset


def get_bal_sampler(dataset):
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("class_bal is only supported for datasets exposing a targets list.")

    ratio = np.bincount(targets)
    weights = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = weights[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataset(opt):
    if opt.isTrain and opt.train_lmdb:
        return LMDBDataset(opt, opt.train_lmdb, is_train=True)
    if (not opt.isTrain) and opt.lmdb_path:
        return LMDBDataset(opt, opt.lmdb_path, is_train=False)
    if (not opt.isTrain) and opt.val_lmdb:
        return LMDBDataset(opt, opt.val_lmdb, is_train=False)
    return RealFakeDataset(opt)


def create_dataloader(opt, preprocess=None):
    dataset = create_dataset(opt)

    if "2b" in opt.arch:
        dataset.transform = preprocess

    if opt.class_bal and isinstance(dataset, LMDBDataset):
        raise ValueError("class_bal is not supported for LMDB datasets.")

    shuffle = bool(opt.isTrain and not opt.serial_batches and not opt.class_bal)
    sampler = None
    if opt.distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    elif opt.class_bal:
        sampler = get_bal_sampler(dataset)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
        pin_memory=torch.cuda.is_available(),
    )
