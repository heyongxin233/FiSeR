import os
import sys
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode
import torch
from torch.utils.data import Dataset
from typing import Any

import lmdb
import six
import ujson as json
import pickle
import zlib

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robustness_utils import apply_image_corruption

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def maybe_apply_eval_corruption(img, opt, is_train: bool):
    if is_train:
        return img
    return apply_image_corruption(
        image=img,
        corruption_type=getattr(opt, "corruption_type", "none"),
        corruption_value=getattr(opt, "corruption_value", 0.0),
        crop_mode=getattr(opt, "crop_mode", "center"),
    )


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
        if random.random() > self.p:
            return tensor

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

def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
        rotation_func = transforms.RandomRotation(180)
        jitter_func = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        mask_func = RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
        rotation_func = transforms.Lambda(lambda img: img)
        jitter_func = transforms.Lambda(lambda img: img)
        mask_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)
        rotation_func = transforms.Lambda(lambda img: img)
        jitter_func = transforms.Lambda(lambda img: img)
        mask_func = transforms.Lambda(lambda img: img)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.cropSize, opt.cropSize))
        # rz_func = transforms.CenterCrop(opt.cropSize)

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                transforms.Lambda(lambda img: maybe_apply_eval_corruption(img, opt, opt.isTrain)),
                rz_func,
                # transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                # rotation_func,
                # jitter_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                mask_func
            ]))
    return dset


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


class LMDBReader:
    def __init__(self, path: str, decode: bool = True, mode: str = "PIL"):
        normalized_path = path.rstrip(os.sep)
        lmdb_kwargs = dict(
            max_readers=100,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if os.path.isfile(normalized_path):
            lmdb_kwargs["subdir"] = False
        env = lmdb.open(normalized_path, **lmdb_kwargs)
        self.txn = env.begin(write=False)
        try:
            self.num_samples = int(self.txn.get("num-samples".encode()))
        except Exception:
            raise ValueError(f"{path} has not num-samples key.")

        self.decode = decode
        self.mode = mode

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx += 1
        image_key = b"i-%09d" % idx
        label_key = b"l-%09d" % idx
        image_enc = self.txn.get(image_key)
        label_enc = self.txn.get(label_key)
        if self.decode:
            return self.image_decode(image_enc, self.mode), self.label_decode(label_enc)
        return image_enc, label_enc

    def get_label(self, idx):
        return self.label_decode(self.txn.get(b"l-%09d" % (idx + 1)))

    @staticmethod
    def image_decode(image_enc, mode="NUMPY"):
        def _decode(enc):
            if mode == "NUMPY":
                imgdata = np.frombuffer(enc, dtype="uint8")
                return cv2.imdecode(imgdata, 1)
            if mode == "PIL":
                return Image.open(BytesIO(enc))
            raise ValueError(f"Unsupported decode mode {mode}")

        try:
            images_enc = pickle.loads(image_enc)
            return [_decode(enc) for enc in images_enc]
        except pickle.UnpicklingError:
            return _decode(image_enc)

    @staticmethod
    def label_decode(label_enc):
        return json.loads(zlib.decompress(label_enc).decode("utf-8"))


def ensure_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, bytes):
        return Image.open(BytesIO(img))
    raise TypeError(f"Unsupported image type: {type(img)}")


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}


def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])


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
            transforms.Lambda(lambda img: maybe_apply_eval_corruption(img, opt, is_train)),
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
        if isinstance(label_info, dict):
            label = int(label_info.get("label", label_info.get("binary_label", 0)))
        else:
            label = int(label_info)
        image = ensure_pil(image).convert('RGB')
        return self.transform(image), torch.tensor(label, dtype=torch.long)

def data_augment(img, opt):
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
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


# rz_dict = {'bilinear': Image.BILINEAR,
           # 'bicubic': Image.BICUBIC,
           # 'lanczos': Image.LANCZOS,
           # 'nearest': Image.NEAREST}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize,opt.loadSize), interpolation=rz_dict[interp])
