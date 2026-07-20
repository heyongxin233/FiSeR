# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io, os, pdb
import cv2, math, random
import numpy as np

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from .lmdb_utils import LMDBReader

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RandomJPEG():
    def __init__(self, quality=95, interval=1, p=0.1):
        if isinstance(quality, tuple):
            self.quality = [i for i in range(quality[0], quality[1]) if i % interval == 0]
        else:
            self.quality = quality
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(self.quality, list):
                quality = random.choice(self.quality)
            else:
                quality = self.quality
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        return img


class RandomGaussianBlur():
    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=1.0):
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.blur(img)
        return img


class RandomMask(object):
    def __init__(self, ratio=0.5, patch_size=16, p=0.5):
        """
        Args:
            ratio (float or tuple of float): If float, the ratio of the image to be masked.
                                             If tuple of float, random sample ratio between the two values.
            patch_size (int): the size of the mask (d*d).
        """
        if isinstance(ratio, float):
            self.fixed_ratio = True
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("Ratio must be a float or a tuple of two floats.")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):

        if random.random() > self.p: return tensor

        _, h, w = tensor.shape
        mask = torch.ones((h, w), dtype=torch.float32)

        if self.fixed_ratio:
            ratio = self.ratio[0]
        else:
            ratio = random.uniform(self.ratio[0], self.ratio[1])

        # Calculate the number of masks needed
        num_masks = int((h * w * ratio) / (self.patch_size ** 2))

        # Generate non-overlapping random positions
        selected_positions = set()
        while len(selected_positions) < num_masks:
            top = random.randint(0, (h // self.patch_size) - 1) * self.patch_size
            left = random.randint(0, (w // self.patch_size) - 1) * self.patch_size
            selected_positions.add((top, left))

        for (top, left) in selected_positions:
            mask[top:top+self.patch_size, left:left+self.patch_size] = 0

        return tensor * mask.expand_as(tensor)


def Get_Transforms(args):

    size = args.input_size

    TRANSFORM_DICT = {
        'resize_BILINEAR': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.BILINEAR),
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.BILINEAR),
            ],
        },

        'resize_NEAREST': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.NEAREST),
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.NEAREST),
            ],
        },

        'crop': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),
            ],
            'eval': [
                transforms.CenterCrop([size, size]),
            ],
        },

        'source': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),
            ],
            'eval': [
            ],
        },
    }

    # region [Augmentations]
    transform_train, transform_eval = TRANSFORM_DICT[args.transform_mode]['train'], TRANSFORM_DICT[args.transform_mode]['eval']

    transform_train.extend([
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(180),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        # RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),
    ])

    transform_eval.append(transforms.ToTensor())
    # endregion

    # region [Perturbatiocns in Testing]
    if args.jpeg_factor is not None:
        transform_eval.insert(0, RandomJPEG(quality=args.jpeg_factor, p=1.0))
    if args.blur_sigma is not None:
        transform_eval.insert(0, transforms.GaussianBlur(kernel_size=5, sigma=args.blur_sigma))
    if args.mask_ratio is not None and args.mask_patch_size is not None:
        transform_eval.append(RandomMask(ratio=args.mask_ratio, patch_size=args.mask_patch_size, p=1.0))
    # endregion

    return transforms.Compose(transform_train), transforms.Compose(transform_eval)


class TrainDataset(Dataset):

    def __init__(self, is_train, args):

        TRANSFORM = Get_Transforms(args)
        self.transform = TRANSFORM[0] if is_train else TRANSFORM[1]
        root = args.data_path if is_train else args.eval_data_path
        if root is None:
            raise ValueError("LMDB path must be provided for training or evaluation.")

        self.reader = LMDBReader(root, decode=True, mode="PIL")

    def __len__(self):

        return len(self.reader)

    def __getitem__(self, index):
        
        image, label = self.reader[index]
        if not isinstance(image, Image.Image):
            raise ValueError("LMDB reader must return PIL images for SAFE pipeline")

        image = image.convert('RGB')
        image = self.transform(image)

        if isinstance(label, dict):
            target = label.get('label', label.get('binary_label', label))
        else:
            target = label

        return image, torch.tensor(int(target))
