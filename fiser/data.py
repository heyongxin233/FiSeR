from __future__ import annotations

import json
import math
import pickle
import random
import zlib
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class LMDBReader:
    """Reader for the LMDB layout used by WildFake and the paper benchmarks."""

    def __init__(self, path: str | Path, decode_images: bool = True) -> None:
        self.path = str(path)
        self.decode_images = decode_images
        self._env: lmdb.Environment | None = None
        self._txn: lmdb.Transaction | None = None
        self._length: int | None = None
        self._open()

    def _open(self) -> None:
        self._env = lmdb.open(
            self.path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
        )
        self._txn = self._env.begin(write=False)
        raw_length = self._txn.get(b"num-samples")
        if raw_length is None:
            raise ValueError(f"LMDB is missing the 'num-samples' key: {self.path}")
        self._length = int(raw_length)

    def _ensure_open(self) -> lmdb.Transaction:
        if self._txn is None:
            self._open()
        assert self._txn is not None
        return self._txn

    def close(self) -> None:
        self._txn = None
        if self._env is not None:
            self._env.close()
            self._env = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_env"] = None
        state["_txn"] = None
        return state

    def __len__(self) -> int:
        assert self._length is not None
        return self._length

    @staticmethod
    def decode_label(encoded: bytes) -> dict[str, Any]:
        return json.loads(zlib.decompress(encoded).decode("utf-8"))

    @staticmethod
    def decode_image(encoded: bytes) -> Image.Image:
        try:
            payload = pickle.loads(encoded)
        except (pickle.UnpicklingError, EOFError, ValueError, TypeError):
            payload = None
        if isinstance(payload, (list, tuple)):
            if len(payload) != 1:
                raise ValueError("FiSeR expects one image per LMDB sample")
            encoded = payload[0]
        with Image.open(BytesIO(encoded)) as image:
            return image.convert("RGB")

    def get_label(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        encoded = self._ensure_open().get(b"l-%09d" % (index + 1))
        if encoded is None:
            raise KeyError(f"Missing label at index {index} in {self.path}")
        return self.decode_label(encoded)

    def __getitem__(self, index: int) -> tuple[Image.Image | bytes, dict[str, Any]]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        txn = self._ensure_open()
        encoded_image = txn.get(b"i-%09d" % (index + 1))
        encoded_label = txn.get(b"l-%09d" % (index + 1))
        if encoded_image is None or encoded_label is None:
            raise KeyError(f"Missing image/label pair at index {index} in {self.path}")
        image = self.decode_image(encoded_image) if self.decode_images else encoded_image
        return image, self.decode_label(encoded_label)


def load_source_map(path: str | Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Source map must be a non-empty JSON object: {path}")
    result = {str(name): int(index) for name, index in mapping.items()}
    if len(set(result.values())) != len(result):
        raise ValueError(f"Source IDs must be unique: {path}")
    return result


def derive_source_map(path: str | Path, nature_source: str = "nature") -> dict[str, int]:
    """Build a deterministic source map from an LMDB without decoding images."""
    reader = LMDBReader(path, decode_images=False)
    try:
        sources = {str(reader.get_label(index).get("src", "")) for index in range(len(reader))}
    finally:
        reader.close()
    sources.discard("")
    if nature_source not in sources:
        raise ValueError(f"'{nature_source}' is absent from {path}")
    ordered = [nature_source] + sorted(sources - {nature_source})
    return {name: index for index, name in enumerate(ordered)}


class RandomJPEGCompression:
    def __init__(self, quality_min: int = 25, quality_max: int = 100) -> None:
        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = BytesIO()
        quality = random.randint(self.quality_min, self.quality_max)
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class RandomDownUpResize:
    def __init__(self, scale_min: float = 0.4, scale_max: float = 1.0) -> None:
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.interpolations = [
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.NEAREST,
            Image.Resampling.LANCZOS,
        ]

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = random.uniform(self.scale_min, self.scale_max)
        small_size = (max(16, int(width * scale)), max(16, int(height * scale)))
        first = random.choice(self.interpolations)
        second = random.choice(self.interpolations)
        return image.resize(small_size, first).resize((width, height), second)


class RandomGaussianNoise:
    def __init__(self, sigma_min: float = 0.0, sigma_max: float = 10.0) -> None:
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __call__(self, image: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        if sigma <= 1e-6:
            return image
        values = np.asarray(image).astype(np.float32)
        noise = np.random.normal(0.0, sigma, values.shape).astype(np.float32)
        return Image.fromarray(np.clip(values + noise, 0, 255).astype(np.uint8), mode="RGB")


def build_train_transform(image_size: int) -> Callable[[Image.Image], Image.Image]:
    basic = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                ratio=(0.75, 4.0 / 3.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        ]
    )
    degradation = transforms.Compose(
        [
            transforms.RandomApply([RandomDownUpResize(0.4, 1.0)], p=0.5),
            transforms.RandomApply([RandomJPEGCompression(25, 100)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomApply([RandomGaussianNoise(0.0, 10.0)], p=0.3),
        ]
    )
    return transforms.Compose([basic, degradation])


class FiSeRLMDBDataset(Dataset):
    """Returns image tensors plus paper-consistent binary and generator labels.

    Binary convention in this repository is 0=natural and 1=synthetic. Existing
    LMDB files store the opposite convention, so labels are derived from `src`.
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        processor: Any,
        source_map: dict[str, int] | None = None,
        nature_source: str = "nature",
        train: bool = False,
        image_size: int = 224,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.reader = LMDBReader(lmdb_path, decode_images=True)
        self.processor = processor
        self.source_map = source_map
        self.nature_source = nature_source
        self.transform = build_train_transform(image_size) if train else None
        self.image_size = int(image_size)

        length = len(self.reader)
        if max_samples is not None and max_samples < length:
            generator = random.Random(seed)
            self.indices = sorted(generator.sample(range(length), int(max_samples)))
        else:
            self.indices = list(range(length))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        lmdb_index = self.indices[index]
        image, metadata = self.reader[lmdb_index]
        assert isinstance(image, Image.Image)
        if self.transform is not None:
            image = self.transform(image)
            encoded = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )
        else:
            encoded = self.processor(images=image, return_tensors="pt")

        source = str(metadata.get("src", ""))
        if not source:
            raise ValueError(f"Sample {lmdb_index} has no 'src' field")
        if self.source_map is None:
            source_id = 0 if source == self.nature_source else 1
        else:
            if source not in self.source_map:
                raise ValueError(f"Unknown source '{source}' at LMDB index {lmdb_index}")
            source_id = self.source_map[source]

        sample_id = int(metadata.get("id", lmdb_index))
        binary_label = int(source != self.nature_source)
        pixel_values = encoded["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "label": binary_label,
            "source_id": source_id,
            "source": source,
            "sample_id": sample_id,
            "lmdb_index": lmdb_index,
        }


def padded_distributed_length(length: int, world_size: int) -> int:
    return int(math.ceil(length / max(world_size, 1)) * max(world_size, 1))
