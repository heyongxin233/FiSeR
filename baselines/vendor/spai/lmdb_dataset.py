import json
import os
import pickle
import zlib
from io import BytesIO
from typing import Any, Optional

import albumentations as A
import cv2
import lmdb
import numpy as np
import torch
from albumentations.augmentations.transforms import ImageCompressionType
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def read_num_samples(path: str) -> int:
    normalized_path = path.rstrip(os.sep)
    lmdb_kwargs = {
        "max_readers": 1,
        "readonly": True,
        "lock": False,
        "readahead": False,
        "meminit": False,
    }
    if os.path.isfile(normalized_path):
        lmdb_kwargs["subdir"] = False

    env = lmdb.open(normalized_path, **lmdb_kwargs)
    try:
        with env.begin(write=False) as txn:
            return int(txn.get(b"num-samples"))
    finally:
        env.close()


class LMDBReader:
    def __init__(self, path: str, decode: bool = True, mode: str = "PIL") -> None:
        normalized_path = path.rstrip(os.sep)
        lmdb_kwargs = {
            "max_readers": 128,
            "readonly": True,
            "lock": False,
            "readahead": False,
            "meminit": False,
        }

        if os.path.isfile(normalized_path):
            lmdb_kwargs["subdir"] = False
        elif not os.path.isdir(normalized_path) or not os.path.isfile(os.path.join(normalized_path, "data.mdb")):
            raise FileNotFoundError(
                f"LMDB path `{path}` is neither a data.mdb file nor a directory containing data.mdb."
            )

        self.env = lmdb.open(normalized_path, **lmdb_kwargs)
        self.txn = self.env.begin(write=False)
        self.decode = decode
        self.mode = mode

        try:
            self.num_samples = int(self.txn.get(b"num-samples"))
        except Exception as exc:
            raise ValueError(f"LMDB at `{path}` is missing `num-samples`.") from exc

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        idx += 1
        image_key = b"i-%09d" % idx
        label_key = b"l-%09d" % idx
        image_bytes = self.txn.get(image_key)
        label_bytes = self.txn.get(label_key)

        if image_bytes is None or label_bytes is None:
            raise IndexError(f"Missing sample {idx} in LMDB.")

        if self.decode:
            return self.image_decode(image_bytes, self.mode), self.label_decode(label_bytes)
        return image_bytes, label_bytes

    @staticmethod
    def image_decode(image_enc: bytes, mode: str = "PIL"):
        def _decode(encoded: bytes):
            if mode == "NUMPY":
                imgdata = np.frombuffer(encoded, dtype="uint8")
                return cv2.imdecode(imgdata, 1)
            if mode == "PIL":
                return Image.open(BytesIO(encoded))
            raise ValueError(f"Unsupported decode mode: {mode}")

        try:
            images = pickle.loads(image_enc)
            return [_decode(item) for item in images]
        except Exception:
            return _decode(image_enc)

    @staticmethod
    def label_decode(label_enc: bytes):
        return json.loads(zlib.decompress(label_enc).decode("utf-8"))


def ensure_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]
        return Image.fromarray(image)
    if isinstance(image, bytes):
        return Image.open(BytesIO(image))
    raise TypeError(f"Unsupported image type: {type(image)}")


def build_transform(is_train: bool, config):
    if is_train:
        transforms_list = []
        if config.AUG.MIN_CROP_AREA == config.AUG.MAX_CROP_AREA:
            transforms_list.append(
                A.PadIfNeeded(min_height=config.DATA.IMG_SIZE, min_width=config.DATA.IMG_SIZE)
            )
            transforms_list.append(
                A.RandomCrop(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE)
            )
        else:
            transforms_list.append(
                A.RandomResizedCrop(
                    size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    scale=(config.AUG.MIN_CROP_AREA, config.AUG.MAX_CROP_AREA),
                )
            )
        transforms_list.extend(
            [
                A.HorizontalFlip(p=config.AUG.HORIZONTAL_FLIP_PROB),
                A.VerticalFlip(p=config.AUG.VERTICAL_FLIP_PROB),
                A.Rotate(
                    limit=config.AUG.ROTATION_DEGREES,
                    crop_border=True,
                    p=config.AUG.ROTATION_PROB,
                ),
            ]
        )
        if config.AUG.ROTATION_PROB > 0.0:
            transforms_list.append(
                A.Resize(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE)
            )
        transforms_list.extend(
            [
                A.GaussianBlur(
                    blur_limit=(3, 9),
                    sigma_limit=(0.01, 0.5),
                    p=config.AUG.GAUSSIAN_BLUR_PROB,
                ),
                A.GaussNoise(p=config.AUG.GAUSSIAN_NOISE_PROB),
                A.ColorJitter(
                    p=config.AUG.COLOR_JITTER,
                    brightness=config.AUG.COLOR_JITTER_BRIGHTNESS_RANGE,
                    contrast=config.AUG.COLOR_JITTER_CONTRAST_RANGE,
                    saturation=config.AUG.COLOR_JITTER_SATURATION_RANGE,
                    hue=config.AUG.COLOR_JITTER_HUE_RANGE,
                ),
                A.Sharpen(
                    p=config.AUG.SHARPEN_PROB,
                    alpha=config.AUG.SHARPEN_ALPHA_RANGE,
                    lightness=config.AUG.SHARPEN_LIGHTNESS_RANGE,
                ),
                A.ImageCompression(
                    quality_lower=config.AUG.JPEG_MIN_QUALITY,
                    quality_upper=config.AUG.JPEG_MAX_QUALITY,
                    compression_type=ImageCompressionType.JPEG,
                    p=config.AUG.JPEG_COMPRESSION_PROB,
                ),
                A.ImageCompression(
                    quality_lower=config.AUG.WEBP_MIN_QUALITY,
                    quality_upper=config.AUG.WEBP_MAX_QUALITY,
                    compression_type=ImageCompressionType.WEBP,
                    p=config.AUG.WEBP_COMPRESSION_PROB,
                ),
            ]
        )
    else:
        transforms_list = [
            A.ImageCompression(
                quality_lower=config.TEST.JPEG_QUALITY,
                quality_upper=config.TEST.JPEG_QUALITY,
                compression_type=ImageCompressionType.JPEG,
                p=1.0 if config.TEST.JPEG_COMPRESSION else 0.0,
            ),
            A.ImageCompression(
                quality_lower=config.TEST.WEBP_QUALITY,
                quality_upper=config.TEST.WEBP_QUALITY,
                compression_type=ImageCompressionType.WEBP,
                p=1.0 if config.TEST.WEBP_COMPRESSION else 0.0,
            ),
            A.GaussianBlur(
                blur_limit=(
                    config.TEST.GAUSSIAN_BLUR_KERNEL_SIZE,
                    config.TEST.GAUSSIAN_BLUR_KERNEL_SIZE,
                ),
                sigma_limit=0,
                p=1.0 if config.TEST.GAUSSIAN_BLUR else 0.0,
            ),
            A.GaussNoise(
                var_limit=(
                    config.TEST.GAUSSIAN_NOISE_SIGMA ** 2,
                    config.TEST.GAUSSIAN_NOISE_SIGMA ** 2,
                ),
                p=1.0 if config.TEST.GAUSSIAN_NOISE else 0.0,
            ),
            A.RandomScale(
                scale_limit=(config.TEST.SCALE_FACTOR - 1, config.TEST.SCALE_FACTOR - 1),
                p=1.0 if config.TEST.SCALE else 0.0,
            ),
        ]
        if config.TEST.MAX_SIZE is not None:
            transforms_list.append(A.SmallestMaxSize(max_size=config.TEST.MAX_SIZE))

        if config.TEST.ORIGINAL_RESOLUTION:
            transforms_list.append(
                A.PadIfNeeded(min_height=config.DATA.IMG_SIZE, min_width=config.DATA.IMG_SIZE)
            )
        elif config.TEST.CROP:
            transforms_list.append(
                A.PadIfNeeded(min_height=config.DATA.IMG_SIZE, min_width=config.DATA.IMG_SIZE)
            )
            transforms_list.append(
                A.CenterCrop(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE)
            )
        else:
            transforms_list.append(A.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))

    if config.MODEL.REQUIRED_NORMALIZATION == "imagenet":
        transforms_list.append(A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    elif config.MODEL.REQUIRED_NORMALIZATION == "positive_0_1":
        transforms_list.append(A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)))
    else:
        raise RuntimeError(f"Unsupported normalization: {config.MODEL.REQUIRED_NORMALIZATION}")

    transforms_list.append(ToTensorV2())
    return A.Compose(transforms_list)


class LmdbSampleDataset(Dataset):
    def __init__(self, lmdb_path: str, config, is_train: bool) -> None:
        self.lmdb_path = lmdb_path
        self.config = config
        self.is_train = is_train
        self.transform = build_transform(is_train=is_train, config=config)
        self.reader: Optional[LMDBReader] = None
        self.num_samples = read_num_samples(self.lmdb_path)

    def _get_reader(self) -> LMDBReader:
        if self.reader is None:
            cv2.setNumThreads(1)
            self.reader = LMDBReader(self.lmdb_path, decode=True, mode="PIL")
        return self.reader

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        image, label_info = self._get_reader()[index]
        image = ensure_pil(image).convert("RGB")
        image_np = np.array(image)
        tensor = self.transform(image=image_np)["image"]
        label = parse_label(label_info)
        return tensor, torch.tensor(label, dtype=torch.float32), torch.tensor(index, dtype=torch.long)


def parse_label(label_info: Any) -> float:
    if isinstance(label_info, dict):
        for key in ("label", "binary_label", "target"):
            if key in label_info:
                return float(label_info[key])
    return float(label_info)


def use_arbitrary_resolution_eval(config, is_train: bool) -> bool:
    return (not is_train) and config.MODEL.RESOLUTION_MODE == "arbitrary" and config.TEST.ORIGINAL_RESOLUTION


def arbitrary_resolution_collate(batch):
    images = [sample[0].unsqueeze(0) for sample in batch]
    labels = torch.stack([sample[1] for sample in batch], dim=0)
    indices = torch.stack([sample[2] for sample in batch], dim=0)
    return images, labels, indices
