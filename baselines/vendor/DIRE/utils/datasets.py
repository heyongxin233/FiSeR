import os
import sys
from io import BytesIO
from random import choice, random

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from utils.config import CONFIGCLASS
from utils.lmdb_utils import LMDBReader, ensure_pil

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robustness_utils import apply_image_corruption


def dataset_folder(root: str, cfg: CONFIGCLASS):
    if cfg.mode == "binary":
        return binary_dataset(root, cfg)
    if cfg.mode == "filename":
        return FileNameDataset(root, cfg)
    raise ValueError("cfg.mode needs to be binary or filename.")


def maybe_apply_eval_corruption(img: Image.Image, cfg: CONFIGCLASS, is_train: bool):
    if is_train:
        return img
    return apply_image_corruption(
        image=img,
        corruption_type=getattr(cfg, "corruption_type", "none"),
        corruption_value=getattr(cfg, "corruption_value", 0.0),
        crop_mode=getattr(cfg, "crop_mode", "center"),
    )


def binary_dataset(root: str, cfg: CONFIGCLASS):
    identity_transform = transforms.Lambda(lambda img: img)

    if cfg.isTrain or cfg.aug_resize:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg))
    else:
        rz_func = identity_transform

    if cfg.isTrain:
        crop_func = transforms.RandomCrop(cfg.cropSize)
    else:
        crop_func = transforms.CenterCrop(cfg.cropSize) if cfg.aug_crop else identity_transform

    if cfg.isTrain and cfg.aug_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = identity_transform

    return datasets.ImageFolder(
        root,
        transforms.Compose(
            [
                transforms.Lambda(lambda img: maybe_apply_eval_corruption(img, cfg, cfg.isTrain)),
                rz_func,
                transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if cfg.aug_norm
                else identity_transform,
            ]
        )
    )


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def _is_lmdb_path(path: str) -> bool:
    if not path:
        return False
    normalized = path.rstrip(os.sep)
    if normalized.endswith(".lmdb"):
        return True
    if os.path.isfile(normalized):
        return True
    return os.path.isfile(os.path.join(normalized, "data.mdb"))


class LMDBDataset(Dataset):
    def __init__(self, cfg: CONFIGCLASS, lmdb_path: str, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train
        self.reader = LMDBReader(lmdb_path, decode=True, mode="PIL")

        identity_transform = transforms.Lambda(lambda img: img)

        if is_train or cfg.aug_resize:
            rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg))
        else:
            rz_func = identity_transform

        if is_train:
            crop_func = transforms.RandomCrop(cfg.cropSize)
        else:
            crop_func = transforms.CenterCrop(cfg.cropSize) if cfg.aug_crop else identity_transform

        if is_train and cfg.aug_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = identity_transform

        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: maybe_apply_eval_corruption(img, cfg, is_train)),
                rz_func,
                transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if cfg.aug_norm
                else identity_transform,
            ]
        )

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        image, label_info = self.reader[idx]
        if isinstance(image, list):
            image = image[0]
        if isinstance(label_info, dict):
            label = int(label_info.get("label", label_info.get("binary_label", 0)))
        else:
            label = int(label_info)
        image = ensure_pil(image).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def blur_jpg_augment(img: Image.Image, cfg: CONFIGCLASS):
    img: np.ndarray = np.array(img)
    if cfg.isTrain:
        if random() < cfg.blur_prob:
            sig = sample_continuous(cfg.blur_sig)
            gaussian_blur(img, sig)

        if random() < cfg.jpg_prob:
            method = sample_discrete(cfg.jpg_method)
            qual = sample_discrete(cfg.jpg_qual)
            img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s: list):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s: list):
    return s[0] if len(s) == 1 else choice(s)


def gaussian_blur(img: np.ndarray, sigma: float):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img: np.ndarray, compress_val: int):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img: np.ndarray, compress_val: int, key: str) -> np.ndarray:
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img: Image.Image, cfg: CONFIGCLASS) -> Image.Image:
    interp = sample_discrete(cfg.rz_interp)
    return TF.resize(img, cfg.loadSize, interpolation=rz_dict[interp])


def get_dataset(cfg: CONFIGCLASS):
    dset_lst = []
    for dataset in cfg.datasets:
        root = os.path.join(cfg.dataset_root, dataset)
        dset = dataset_folder(root, cfg)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset: torch.utils.data.ConcatDataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def get_dataset_with_lmdb(cfg: CONFIGCLASS, lmdb_path: str = None, is_train: bool = True):
    if lmdb_path or _is_lmdb_path(getattr(cfg, "train_lmdb", None)) or _is_lmdb_path(getattr(cfg, "val_lmdb", None)):
        target_path = lmdb_path or (cfg.train_lmdb if is_train else cfg.val_lmdb)
        if not target_path:
            raise ValueError("A valid lmdb path must be provided for LMDB training.")
        return LMDBDataset(cfg, target_path, is_train=is_train)
    return get_dataset(cfg)


def create_dataloader(cfg: CONFIGCLASS, lmdb_path: str = None, is_train: bool = True, distributed: bool = False):
    shuffle = not cfg.serial_batches if (is_train and not cfg.class_bal) else False
    dataset = get_dataset_with_lmdb(cfg, lmdb_path=lmdb_path, is_train=is_train)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    elif cfg.class_bal and isinstance(dataset, torch.utils.data.ConcatDataset):
        sampler = get_bal_sampler(dataset)
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
    )
