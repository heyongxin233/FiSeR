from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from bit_patch import bit_patch as bit_patch_process
from lmdb_utils import LMDBReader


def create_preprocessing_pipeline(options):
    if options.use_patch:
        transform_func = transforms.Lambda(
            lambda img: bit_patch_process(
                img,
                options.img_height,
                options.bit_mode,
                options.patch_size,
                options.patch_mode,
            )
        )
    else:
        transform_func = transforms.Resize((options.img_height, options.img_height))

    return transforms.Compose(
        [
            transform_func,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def ensure_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, bytes):
        return Image.open(BytesIO(img))
    raise TypeError(f'Unsupported image type: {type(img)}')


class LMDBImageDataset(Dataset):
    def __init__(self, lmdb_path: str, options):
        self.options = options
        self.reader = LMDBReader(lmdb_path, decode=True, mode='PIL')
        self.transform = create_preprocessing_pipeline(options)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        image, label_info = self.reader[index]
        image = ensure_pil(image).convert('RGB')
        if isinstance(label_info, dict):
            label = float(label_info.get('label', label_info.get('binary_label', 0)))
        else:
            label = float(label_info)
        return self.transform(image), torch.tensor(label, dtype=torch.float32)


def create_dataloader(options, lmdb_path, is_train, distributed=False):
    if not lmdb_path:
        raise ValueError('A valid LMDB path is required.')

    dataset = LMDBImageDataset(lmdb_path, options)
    sampler = None
    shuffle = is_train
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=is_train,
            num_replicas=options.world_size,
            rank=options.rank,
            drop_last=False,
        )
        shuffle = False

    batch_size = options.batch_size if is_train else options.val_batch_size
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=options.num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=options.num_workers > 0,
    )
    return data_loader, sampler
