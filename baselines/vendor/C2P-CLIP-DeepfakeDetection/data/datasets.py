import json
import math
import os
import pickle
import random
import re
import zlib
from io import BytesIO
from typing import Any, Callable, Optional, Tuple

import lmdb
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils.util import resolve_hf_pretrained_path

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

REAL_HINTS = {"real", "nature", "camera", "photograph", "photo"}
GENERIC_TOKENS = {
    "images",
    "image",
    "image detection datasets",
    "imgs",
    "img",
    "train",
    "test",
    "val",
    "valid",
    "validation",
    "typical",
    "advanced",
    "personalizedsd",
    "temp4",
    "temp45",
    "new",
    "wildfake",
    "data",
}


try:
    from torchvision.transforms import InterpolationMode

    RZ_DICT = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
        "nearest": InterpolationMode.NEAREST,
    }
except ImportError:
    RZ_DICT = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
        "nearest": Image.NEAREST,
    }


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class LMDBReader:
    def __init__(self, path: str, decode: bool = True, mode: str = "PIL"):
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
        self.decode = decode
        self.mode = mode

        num_samples = self.txn.get(b"num-samples")
        if num_samples is None:
            raise ValueError(f"LMDB at {path} is missing num-samples.")
        self.num_samples = int(num_samples)

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

    @staticmethod
    def image_decode(image_enc, mode="PIL"):
        def _decode(enc):
            if mode == "PIL":
                return Image.open(BytesIO(enc))
            if mode == "NUMPY":
                import cv2

                imgdata = np.frombuffer(enc, dtype="uint8")
                return cv2.imdecode(imgdata, 1)
            raise ValueError(f"Unsupported decode mode: {mode}")

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


def _sanitize_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_scene(path: str) -> str:
    if not path:
        return ""

    parts = path.replace("\\", "/").split("/")
    candidates = []
    for part in reversed(parts[:-1]):
        token = _sanitize_text(os.path.splitext(part)[0])
        low = token.lower()
        if not token or token == "." or low in GENERIC_TOKENS:
            continue
        if low in {"real", "fake", "gan based", "diffusion based", "other based"}:
            continue
        if re.fullmatch(r"[0-9a-f]{8,}", low):
            continue
        if token.lower().startswith("img") and any(ch.isdigit() for ch in token):
            continue
        candidates.append(token)
        if len(candidates) == 2:
            break

    candidates.reverse()
    deduped = []
    for token in candidates:
        if not deduped or deduped[-1].lower() != token.lower():
            deduped.append(token)
    return " ".join(deduped)


def _extract_raw_label(label_info: Any) -> int:
    if isinstance(label_info, dict):
        for key in ("binary_label", "label", "target"):
            if key in label_info:
                return int(label_info[key])
    return int(label_info)


def normalize_label(label_info: Any, opt) -> int:
    if isinstance(label_info, dict):
        generator = str(label_info.get("Generator", "")).strip().lower()
        src = str(label_info.get("src", "")).strip().lower()
        path = str(label_info.get("path", "")).strip().lower()

        if generator in REAL_HINTS or src in REAL_HINTS:
            return 0
        if "/0_real/" in path or "/real/" in path:
            return 0
        if "/1_fake/" in path or "/fake/" in path:
            return 1

    raw_label = _extract_raw_label(label_info)
    return 0 if raw_label == getattr(opt, "lmdb_real_label", 1) else 1


def build_prompt(label_info: Any, target: int, opt) -> str:
    if isinstance(label_info, dict):
        path = str(label_info.get("path", ""))
        src = _sanitize_text(label_info.get("src", ""))
        generator = _sanitize_text(label_info.get("Generator", ""))
    else:
        path = ""
        src = ""
        generator = ""

    scene = _extract_scene(path)
    cates = list(getattr(opt, "cates", []))
    split = max(1, len(cates) // 2)
    fake_common = " ".join(cates[:split]) if cates else "Deepfake"
    real_common = " ".join(cates[split:]) if len(cates) > split else "Camera"

    if target == 1:
        phrases = ["AI generated image"]
        if src and src.lower() not in REAL_HINTS:
            phrases.append(f"from {src}")
        if generator and generator.lower() != "real":
            phrases.append(f"with {generator} generation")
        if scene and src.lower() not in scene.lower():
            phrases.append(f"showing {scene}")
        detail = " ".join(phrases)
        common = fake_common
    else:
        phrases = ["real camera photo"]
        if scene:
            phrases.append(f"of {scene}")
        elif src:
            phrases.append(f"from {src}")
        detail = " ".join(phrases)
        common = real_common

    return f"{common}. {detail}. {common}."


def tokenize_text(tokenizer, text: str):
    inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs["input_ids"][0], inputs["attention_mask"][0]


class ImageFolder2(datasets.DatasetFolder):
    def __init__(self, root: str, opt, transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, extensions=IMG_EXTENSIONS, loader=pil_loader)
        self.opt = opt
        self.clip_name = resolve_hf_pretrained_path(self.opt.clip)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.clip_name,
            model_max_length=77,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token_id = 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        text = None
        if getattr(self.opt, "textroot", None):
            textpath = path.replace(self.opt.imgroot, self.opt.textroot)
            textpath = os.path.splitext(textpath)[0] + ".txt"
            if os.path.exists(textpath):
                with open(textpath, "r", encoding="utf-8", errors="ignore") as file:
                    text = file.read().strip()

        if not text:
            text = build_prompt({"path": path, "src": "", "Generator": "Real" if target == 0 else "Fake"}, target, self.opt)
        else:
            cates_len = max(1, len(self.opt.cates) // 2)
            fake_common = " ".join(self.opt.cates[:cates_len])
            real_common = " ".join(self.opt.cates[cates_len:])
            common = fake_common if target == 1 else real_common
            text = f"{common}. {text} {common}."

        input_ids, attention_mask = tokenize_text(self.tokenizer, text)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, text, input_ids, attention_mask, target


class LMDBDataset(Dataset):
    def __init__(self, opt, lmdb_path: str, is_train: bool):
        self.opt = opt
        self.is_train = is_train
        self.lmdb_path = lmdb_path
        self.reader = LMDBReader(lmdb_path, decode=True, mode="PIL")
        self.clip_name = resolve_hf_pretrained_path(self.opt.clip)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.clip_name,
            model_max_length=77,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token_id = 0

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

        if (not is_train) and getattr(opt, "no_resize", False):
            resize_func = transforms.Lambda(lambda img: img)
        else:
            resize_func = transforms.Lambda(lambda img: translate_duplicate(img, opt.cropSize))

        transform_steps = [resize_func]
        if getattr(opt, "data_aug", False):
            transform_steps.append(transforms.Lambda(lambda img: data_augment(img, opt)))
        transform_steps.extend(
            [
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        image, label_info = self.reader[idx]
        image = ensure_pil(image).convert("RGB")

        target = normalize_label(label_info, self.opt)
        text = build_prompt(label_info, target, self.opt)
        input_ids, attention_mask = tokenize_text(self.tokenizer, text)

        path = f"{self.lmdb_path}:{idx}"
        if isinstance(label_info, dict):
            path = str(label_info.get("path", path))

        return path, self.transform(image), text, input_ids, attention_mask, target


def dataset_folder(opt, root):
    if opt.mode == "binary":
        return binary_dataset(opt, root)
    if opt.mode == "filename":
        return FileNameDataset(opt, root)
    raise ValueError("opt.mode needs to be binary or filename.")


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif getattr(opt, "no_crop", False):
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)

    if (not opt.isTrain) and getattr(opt, "no_resize", False):
        resize_func = transforms.Lambda(lambda img: img)
    else:
        resize_func = transforms.Lambda(lambda img: translate_duplicate(img, opt.cropSize))

    transform_steps = [resize_func]
    if getattr(opt, "data_aug", False):
        transform_steps.append(transforms.Lambda(lambda img: data_augment(img, opt)))
    transform_steps.extend(
        [
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    return ImageFolder2(root, opt, transforms.Compose(transform_steps))


class FileNameDataset(datasets.ImageFolder):
    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        return path


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


def data_augment(img, opt):
    img = np.array(img)

    if random.random() < opt.blur_prob:
        sigma = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sigma)

    if random.random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        quality = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, quality, method)

    return Image.fromarray(img)


def sample_continuous(values):
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return random.random() * (values[1] - values[0]) + values[0]
    raise ValueError("Length of iterable should be 1 or 2.")


def sample_discrete(values):
    if len(values) == 1:
        return values[0]
    return random.choice(values)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    import cv2

    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    _, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    Image.fromarray(img).save(out, format="jpeg", quality=compress_val)
    pil_img = Image.open(out)
    array = np.array(pil_img)
    out.close()
    return array


JPEG_DICT = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img, compress_val, key):
    return JPEG_DICT[key](img, compress_val)


def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=RZ_DICT[interp])
