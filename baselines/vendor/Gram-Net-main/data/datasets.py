import random
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.lmdb_utils import LMDBReader

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LMDBDataset(Dataset):
    def __init__(self, opt, lmdb_path: str, is_train: bool):
        self.opt = opt
        self.is_train = is_train
        self.reader = LMDBReader(lmdb_path, decode=True, mode="PIL")

        if is_train:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if is_train and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)

        if (not is_train) and getattr(opt, 'no_resize', False):
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

        use_aug = getattr(opt, 'data_aug', False) or is_train
        self.transform = transforms.Compose([
            rz_func,
            transforms.Lambda(lambda img: data_augment(img, opt) if use_aug else img),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        image, label_info = self.reader[idx]
        if isinstance(image, list):
            image = image[0]
        label = extract_label(label_info)
        image = ensure_pil(image)
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def ensure_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, bytes):
        return Image.open(BytesIO(img))
    raise TypeError(f"Unsupported image type: {type(img)}")


def extract_label(label_info: Any) -> int:
    if isinstance(label_info, dict):
        return int(label_info.get("label", label_info.get("binary_label", 0)))
    return int(label_info)


def data_augment(img: Image.Image, opt) -> Image.Image:
    img = np.array(img)

    if random.random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random.random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    import cv2

    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    _, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}


def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
