import torch
from torch.utils.data.distributed import DistributedSampler

from .datasets import LMDBDataset


def create_dataloader(opt, lmdb_path=None, is_train=None, distributed=None):
    is_train = opt.isTrain if is_train is None else is_train
    distributed = opt.distributed if distributed is None else distributed
    lmdb_target = lmdb_path if lmdb_path is not None else opt.lmdb_path
    if not lmdb_target:
        raise ValueError('lmdb_path must be provided for loading data')

    shuffle = not opt.serial_batches if is_train else False
    dataset = LMDBDataset(opt, lmdb_target, is_train=is_train)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, num_replicas=opt.world_size, rank=opt.rank, drop_last=False)
        shuffle = False

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads),
                                              pin_memory=True)
    return data_loader, sampler
