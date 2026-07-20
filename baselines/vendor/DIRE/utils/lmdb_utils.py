import os
from io import BytesIO
import pickle
from typing import Any

import cv2
import lmdb
import numpy as np
import ujson as json
import zlib
from PIL import Image


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
