# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import lmdb
import os
import pickle
import sys
import zlib
from io import BytesIO

import cv2

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dct import DCT_base_Rec_Module

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robustness_utils import apply_image_corruption

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),
])

transform_train = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def pil_collate(batch):
    images, targets = zip(*batch)
    targets = torch.stack(targets)
    return list(images), targets


class LMDBReader:
    def __init__(self, path, decode=True, mode="PIL"):
        self.decode = decode
        self.mode = mode

        # Support both LMDB directory paths (containing data.mdb/lock.mdb) and
        # direct paths to data.mdb files. The latter would otherwise trigger
        # "lock.mdb: Not a directory" errors when the provided path ends with
        # the file instead of its parent folder. We also handle trailing
        # slashes on file paths, which can make os.path.isfile return False.
        normalized_path = path.rstrip(os.sep)
        lmdb_path = normalized_path
        lmdb_kwargs = {
            "max_readers": 100,
            "readonly": True,
            "lock": False,
            "readahead": False,
            "meminit": False,
        }

        if os.path.isfile(normalized_path):
            lmdb_kwargs["subdir"] = False
        elif not os.path.isdir(normalized_path) or not os.path.isfile(os.path.join(normalized_path, "data.mdb")):
            raise FileNotFoundError(
                f"提供的 LMDB 路径 {path} 既不是 data.mdb 文件，也不是包含 data.mdb 的目录。"
            )

        env = lmdb.open(lmdb_path, **lmdb_kwargs)
        self.txn = env.begin(write=False)
        try:
            self.num_samples = int(self.txn.get('num-samples'.encode()))
        except Exception:
            raise ValueError(f"LMDB at {path} missing num-samples key")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx += 1
        image_key = b'i-%09d' % idx
        label_key = b'l-%09d' % idx
        image_enc = self.txn.get(image_key)
        label_enc = self.txn.get(label_key)
        if self.decode:
            return self.image_decode(image_enc, self.mode), self.label_decode(label_enc)
        return image_enc, label_enc

    @staticmethod
    def image_decode(image_enc, mode="PIL"):
        def _decode(enc):
            if mode == "NUMPY":
                imgdata = np.frombuffer(enc, dtype='uint8')
                return cv2.imdecode(imgdata, 1)
            if mode == "PIL":
                return Image.open(BytesIO(enc))
            raise ValueError(f"Unsupported decode mode: {mode}")

        try:
            images_enc = pickle.loads(image_enc)
            return [_decode(enc) for enc in images_enc]
        except pickle.UnpicklingError:
            return _decode(image_enc)

    @staticmethod
    def label_decode(label_enc):
        return json.loads(zlib.decompress(label_enc).decode("utf-8"))


class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        root = args.data_path if is_train else args.eval_data_path
        self.reader = LMDBReader(root, decode=True, mode="PIL")
        self.is_train = is_train

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        image, label = self.reader[index]
        if not isinstance(image, Image.Image):
            raise ValueError("LMDB reader must return PIL images for AIDE pipeline")

        image = image.convert('RGB')

        target = int(label.get('label', label))

        return image, torch.tensor(target)

    

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        root = args.data_path if is_train else args.eval_data_path
        print("TestDataset root:", root)
        self.reader = LMDBReader(root, decode=True, mode="PIL")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        image, label = self.reader[index]
        image = image.convert('RGB')

        target = int(label.get('label', label))

        return image, torch.tensor(target)


class OnDeviceAIDEPreprocessor:
    def __init__(
        self,
        device,
        use_perturbations=True,
        corruption_type="none",
        corruption_value=0.0,
        crop_mode="center",
    ):
        self.device = device
        self.use_perturbations = use_perturbations
        self.corruption_type = corruption_type
        self.corruption_value = float(corruption_value)
        self.crop_mode = crop_mode
        self.dct = DCT_base_Rec_Module().to(device)
        self.perturbations = Perturbations.to(device) if use_perturbations else None

    def __call__(self, images):
        # Accept either a batch tensor or a list of PIL/tensors from the dataloader
        if isinstance(images, (list, tuple)):
            resized = []
            for img in images:
                if isinstance(img, Image.Image):
                    img = apply_image_corruption(
                        image=img,
                        corruption_type=self.corruption_type,
                        corruption_value=self.corruption_value,
                        crop_mode=self.crop_mode,
                    )
                    img = transform_to_tensor(img)
                elif not isinstance(img, torch.Tensor):
                    raise TypeError(f"Unsupported image type {type(img)} for preprocessing")

                if img.ndim == 3:
                    img = img.unsqueeze(0)

                img = img.to(self.device, non_blocking=True)
                img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
                resized.append(img.squeeze(0))

            images = torch.stack(resized, dim=0)
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            images = images.to(self.device, non_blocking=True)
            if images.shape[-2:] != (256, 256):
                images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            raise TypeError(f"Unsupported batch type {type(images)} for preprocessing")

        if self.perturbations is not None:
            images = self.perturbations(images)

        processed_batch = []
        for image in images:
            try:
                x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
            except Exception:
                # fall back to a deterministic zero tensor if DCT fails
                zero = torch.zeros_like(image)
                x_minmin = x_maxmax = x_minmin1 = x_maxmax1 = zero

            # Ensure DCT reconstructions match the resized image spatial size
            target_hw = image.shape[-2:]
            resize_fn = lambda x: F.interpolate(x.unsqueeze(0), size=target_hw, mode='bilinear', align_corners=False).squeeze(0)
            x_minmin = resize_fn(x_minmin)
            x_maxmax = resize_fn(x_maxmax)
            x_minmin1 = resize_fn(x_minmin1)
            x_maxmax1 = resize_fn(x_maxmax1)

            x_0 = transform_train(image)
            x_minmin = transform_train(x_minmin)
            x_maxmax = transform_train(x_maxmax)
            x_minmin1 = transform_train(x_minmin1)
            x_maxmax1 = transform_train(x_maxmax1)

            processed_batch.append(torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0))

        return torch.stack(processed_batch, dim=0)

