import torch
from torch.utils.data.distributed import DistributedSampler

from .datasets import LMDBDataset


def create_dataloader(opt, lmdb_path: str, is_train: bool, distributed: bool):
    dataset = LMDBDataset(opt, lmdb_path, is_train=is_train)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)
        shuffle = False
    else:
        sampler = None
        shuffle = is_train and not opt.serial_batches

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
        pin_memory=True,
        drop_last=is_train,
    )
    return data_loader, sampler
