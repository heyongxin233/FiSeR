import os

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import LMDBDataset, dataset_folder


# 获取数据集，要根据不同的需求进行修改
'''
def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)
'''


def get_dataset(opt):
    classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
    # 如果类文件夹中没有 '0_real' 和 '1_fake'，会对每个类单独加载数据集，并将它们合并返回
    # 如果类文件夹中包含 '0_real' 和 '1_fake'，则直接对整个根目录加载数据集
    if '0_real' not in classes or '1_fake' not in classes:
        dset_lst = []
        for cls in classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    return dataset_folder(opt, opt.dataroot)


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


def create_dataloader(opt, lmdb_path=None, is_train=True, distributed=False):
    if lmdb_path:
        dataset = LMDBDataset(opt, lmdb_path=lmdb_path, is_train=is_train)
    else:
        dataset = get_dataset(opt)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
    elif opt.class_bal and is_train:
        sampler = get_bal_sampler(dataset)

    shuffle = sampler is None and (not opt.serial_batches and is_train)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
        drop_last=opt.drop_last,
        pin_memory=True,
    )
    return data_loader, sampler
