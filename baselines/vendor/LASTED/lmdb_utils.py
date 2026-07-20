import json
import lmdb
import os
import pickle
import zlib
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class LMDBReader:
    def __init__(self, path: str, decode: bool = True, mode: str = "PIL"):
        self.decode = decode
        self.mode = mode

        normalized_path = path.rstrip(os.sep)
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
                f"Provided LMDB path {path} is neither a data.mdb file nor a directory containing data.mdb."
            )

        env = lmdb.open(normalized_path, **lmdb_kwargs)
        self.txn = env.begin(write=False)
        try:
            self.num_samples = int(self.txn.get('num-samples'.encode()))
        except Exception as exc:
            raise ValueError(f"LMDB at {path} missing num-samples key") from exc

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


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, data_size: int = 448, is_train: bool = True):
        from albumentations import (
            CenterCrop,
            Compose,
            GaussNoise,
            GaussianBlur,
            HorizontalFlip,
            ImageCompression,
            OneOf,
            PadIfNeeded,
            RandomCrop,
            RandomRotate90,
            ToGray,
            VerticalFlip,
        )
        from torchvision import transforms

        self.reader = LMDBReader(lmdb_path, decode=True, mode="PIL")
        self.data_size = data_size
        self.is_train = is_train

        self.albu_pre_train = Compose([
            PadIfNeeded(
                min_height=self.data_size,
                min_width=self.data_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
            RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            OneOf([
                ImageCompression(quality_lower=50, quality_upper=95, compression_type="jpeg", p=1.0),
                GaussianBlur(blur_limit=(3, 7), p=1.0),
                GaussNoise(var_limit=(3.0, 10.0), p=1.0),
                ToGray(p=1.0),
            ], p=0.5),
            RandomRotate90(p=0.33),
            OneOf([HorizontalFlip(p=1.0), VerticalFlip(p=1.0)], p=0.33),
        ], p=1.0)
        self.albu_pre_val = Compose([
            PadIfNeeded(
                min_height=self.data_size,
                min_width=self.data_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
            CenterCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)
        self.imagenet_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.reader)

    def _extract_label(self, label):
        if isinstance(label, dict):
            for key in ("label", "binary_label", "class", "target"):
                if key in label:
                    return int(label[key])
        return int(label)

    def _transform(self, image_np):
        if self.is_train:
            image_np = self.albu_pre_train(image=image_np)['image']
        else:
            image_np = self.albu_pre_val(image=image_np)['image']
        return self.imagenet_norm(image_np)

    def __getitem__(self, idx):
        image, label = self.reader[idx]
        if isinstance(image, (list, tuple)):
            image = image[0]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            image = image[..., ::-1]
        else:
            raise TypeError(f"Unsupported image type {type(image)} from LMDB.")

        tensor = self._transform(image)
        return tensor, torch.tensor(self._extract_label(label))
