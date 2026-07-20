import json
import math
import os
import pickle
import zlib
from io import BytesIO
from random import choice, random, shuffle

import cv2
import lmdb
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
    "beitv2": [0.485, 0.456, 0.406],
    "siglip": [0.5, 0.5, 0.5],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
    "beitv2": [0.229, 0.224, 0.225],
    "siglip": [0.5, 0.5, 0.5],
}


def translate_duplicate(img, crop_size):
    if min(img.size) < crop_size:
        width, height = img.size

        new_width = width * math.ceil(crop_size / width)
        new_height = height * math.ceil(crop_size / height)

        new_img = Image.new("RGB", (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    return img


def recursively_read(rootdir, must_contain, classes=None, exts=None):
    if classes is None:
        classes = []
    if exts is None:
        exts = ["png", "jpg", "JPEG", "jpeg"]

    out = []
    for current_root, _, files in os.walk(rootdir):
        for file_name in files:
            if "." not in file_name:
                continue
            ext = file_name.rsplit(".", 1)[-1]
            full_path = os.path.join(current_root, file_name)
            if ext in exts and must_contain in full_path:
                if not classes or full_path.split("/")[-3] in classes:
                    out.append(full_path)
    return out


def get_list(path, must_contain="", classes=None):
    if classes is None:
        classes = []

    if path.endswith(".pickle"):
        with open(path, "rb") as handle:
            image_list = pickle.load(handle)
        return [item for item in image_list if must_contain in item]

    return recursively_read(path, must_contain, classes)


def get_stat_source(arch):
    arch = arch.lower()
    if arch.startswith("imagenet"):
        return "imagenet"
    if arch.startswith("clip"):
        return "clip"
    if arch.startswith("siglip"):
        return "siglip"
    if arch.startswith("beitv2"):
        return "beitv2"
    raise ValueError(f"Unsupported arch for normalization stats: {arch}")


def build_transform(opt, is_train):
    should_log = (not getattr(opt, "distributed", False)) or getattr(opt, "local_rank", 0) == 0
    if "2b" in opt.arch:
        if should_log:
            print("Using CLIP 2B transform")
        return None

    if is_train:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif getattr(opt, "no_crop", False):
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if is_train and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)

    use_extra_aug = is_train and getattr(opt, "data_aug", False)
    stat_from = get_stat_source(opt.arch)

    if should_log:
        print("mean and std stats are from:", stat_from)
        print("using Official CLIP's normalization")
    return transforms.Compose([
        transforms.Lambda(lambda img: translate_duplicate(img, opt.loadSize)),
        transforms.Lambda(lambda img: data_augment(img, opt) if use_extra_aug else img),
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
    ])


def ensure_pil(img):
    if isinstance(img, list):
        img = img[0]
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, bytes):
        return Image.open(BytesIO(img))
    raise TypeError(f"Unsupported image type: {type(img)}")


def extract_binary_label(label_info):
    if isinstance(label_info, dict):
        for key in ("label", "binary_label", "target"):
            if key in label_info:
                return int(label_info[key])
        raise KeyError(f"Cannot find a binary label key in LMDB label: {label_info}")
    return int(label_info)


class LMDBReader:
    def __init__(self, path, decode=True, mode="PIL"):
        self.decode = decode
        self.mode = mode
        self.path = path.rstrip(os.sep)

        lmdb_kwargs = {
            "max_readers": 100,
            "readonly": True,
            "lock": False,
            "readahead": False,
            "meminit": False,
        }

        if os.path.isfile(self.path):
            lmdb_kwargs["subdir"] = False
        elif not os.path.isdir(self.path) or not os.path.isfile(os.path.join(self.path, "data.mdb")):
            raise FileNotFoundError(
                f"Provided LMDB path {path} is neither a data.mdb file nor a directory containing data.mdb."
            )

        self.env = lmdb.open(self.path, **lmdb_kwargs)
        self.txn = self.env.begin(write=False)
        raw_num_samples = self.txn.get(b"num-samples")
        if raw_num_samples is None:
            raise ValueError(f"LMDB at {path} is missing the num-samples key")
        self.num_samples = int(raw_num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx += 1
        image_key = f"i-{idx:09d}".encode()
        label_key = f"l-{idx:09d}".encode()
        image_enc = self.txn.get(image_key)
        label_enc = self.txn.get(label_key)
        if image_enc is None or label_enc is None:
            raise IndexError(f"Missing LMDB entry for index {idx} in {self.path}")
        if self.decode:
            return self.image_decode(image_enc, self.mode), self.label_decode(label_enc)
        return image_enc, label_enc

    @staticmethod
    def image_decode(image_enc, mode="PIL"):
        def _decode(enc):
            if mode == "NUMPY":
                imgdata = np.frombuffer(enc, dtype="uint8")
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


class RealFakeDataset(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]

        self.data_label = opt.data_label
        if opt.data_mode == "ours":
            pickle_name = "train.pickle" if opt.data_label == "train" else "val.pickle"
            real_list = get_list(os.path.join(opt.real_list_path, pickle_name))
            fake_list = get_list(os.path.join(opt.fake_list_path, pickle_name))
        elif opt.data_mode == "wang2020":
            folder_name = "train" if opt.data_label == "train" else "test/progan"
            real_list = get_list(os.path.join(opt.wang2020_data_path, folder_name), must_contain="0_real")
            fake_list = get_list(os.path.join(opt.wang2020_data_path, folder_name), must_contain="1_fake")
        elif opt.data_mode == "ours_wang2020":
            pickle_name = "train.pickle" if opt.data_label == "train" else "val.pickle"
            real_list = get_list(os.path.join(opt.real_list_path, pickle_name))
            fake_list = get_list(os.path.join(opt.fake_list_path, pickle_name))

            folder_name = "train" if opt.data_label == "train" else "test/progan"
            real_list += get_list(os.path.join(opt.wang2020_data_path, folder_name), must_contain="0_real")
            fake_list += get_list(os.path.join(opt.wang2020_data_path, folder_name), must_contain="1_fake")
        else:
            raise ValueError(f"Unsupported data_mode: {opt.data_mode}")

        self.labels_dict = {}
        for path in real_list:
            self.labels_dict[path] = 0
        for path in fake_list:
            self.labels_dict[path] = 1

        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        if getattr(opt, "max_samples", None):
            self.total_list = self.total_list[:opt.max_samples]
        self.targets = [self.labels_dict[path] for path in self.total_list]
        self.transform = build_transform(opt, is_train=opt.isTrain)

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class LMDBDataset(Dataset):
    def __init__(self, opt, lmdb_path, is_train):
        self.reader = LMDBReader(lmdb_path, decode=True, mode="PIL")
        self.transform = build_transform(opt, is_train=is_train)
        self.length = len(self.reader)
        if getattr(opt, "max_samples", None):
            self.length = min(self.length, opt.max_samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image, label_info = self.reader[idx]
        image = ensure_pil(image).convert("RGB")
        label = extract_binary_label(label_info)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        value_range = s[1] - s[0]
        return random() * value_range + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    _, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    image = Image.fromarray(img)
    image.save(out, format="jpeg", quality=compress_val)
    image = Image.open(out)
    image = np.array(image)
    out.close()
    return image


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}


def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
