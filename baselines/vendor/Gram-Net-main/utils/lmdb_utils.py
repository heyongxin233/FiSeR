import json
import os
import pickle
import zlib
from io import BytesIO

import cv2
import lmdb
import numpy as np
from PIL import Image


class LMDBReader:
    def __init__(self, path, decode=True, mode="PIL"):
        self.decode = decode
        self.mode = mode

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
                f"Provided LMDB path {path} is neither a data.mdb file nor a directory containing data.mdb."
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
        except Exception:
            return _decode(image_enc)

    @staticmethod
    def label_decode(label_enc):
        try:
            return json.loads(zlib.decompress(label_enc).decode("utf-8"))
        except Exception:
            try:
                return pickle.loads(label_enc)
            except Exception:
                return int(label_enc)
