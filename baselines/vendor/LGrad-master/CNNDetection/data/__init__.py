import os
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .datasets import dataset_folder, LMDBDataset


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


def _select_data_path(opt, is_train, lmdb_path):
    if lmdb_path:
        return lmdb_path
    if is_train:
        return getattr(opt, 'train_lmdb', None) or opt.dataroot
    return getattr(opt, 'val_lmdb', None) or getattr(opt, 'lmdb_path', None) or opt.dataroot


def get_dataset(opt, is_train=True, lmdb_path=None):
    target_path = _select_data_path(opt, is_train, lmdb_path)
    if target_path is None:
        raise ValueError('A valid lmdb or folder path must be provided for the dataset.')

    is_lmdb = False
    if target_path:
        is_lmdb = target_path.endswith('.lmdb') or os.path.isfile(target_path) or os.path.isfile(os.path.join(target_path, 'data.mdb'))

    if is_lmdb:
        return LMDBDataset(opt, target_path, is_train=is_train)

    classes = os.listdir(target_path) if len(opt.classes) == 0 else opt.classes
    # 如果类文件夹中没有 '0_real' 和 '1_fake'，会对每个类单独加载数据集，并将它们合并返回
    # 如果类文件夹中包含 '0_real' 和 '1_fake'，则直接对整个根目录加载数据集
    if '0_real' not in classes or '1_fake' not in classes:
        dset_lst = []
        for cls in classes:
            root = target_path + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    return dataset_folder(opt, target_path)


def get_bal_sampler(dataset):
    targets = []
    if hasattr(dataset, 'datasets'):
        for d in dataset.datasets:
            targets.extend(d.targets)
    elif hasattr(dataset, 'targets'):
        targets.extend(dataset.targets)
    else:
        return None

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt, lmdb_path=None, is_train=None, distributed=None):
    is_train = opt.isTrain if is_train is None else is_train
    distributed = getattr(opt, 'distributed', False) if distributed is None else distributed

    shuffle = not opt.serial_batches if (is_train and not opt.class_bal) else False
    dataset = get_dataset(opt, is_train=is_train, lmdb_path=lmdb_path)
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
                                              drop_last=opt.drop_last,
                                              pin_memory=True)
    return data_loader, sampler
