import pickle
import zlib
from io import BytesIO

import cv2
import lmdb
import numpy as np
import six
import ujson as json
from PIL import Image

import os
import shutil


def mkdir(path):
    """Create a clean directory at ``path``."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


class LMDBWriter:
    def __init__(self, path, cache_size=1000, map_size=1099511627776):
        mkdir(path)
        self.path = path
        self.env = lmdb.open(path, map_size=map_size)
        self.num_samples = 0
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return self.num_samples

    def add(self, image_enc, label_enc):
        self.num_samples += 1
        image_key = b"i-%09d" % self.num_samples
        label_key = b"l-%09d" % self.num_samples
        if image_enc is not None:
            if isinstance(image_enc, Image.Image):
                image_enc = self.image_encode(image_enc)
            self.cache[image_key] = image_enc
        else:
            print("Warning: image is None")
        if isinstance(label_enc, dict):
            label_enc = self.label_encode(label_enc)
        self.cache[label_key] = label_enc
        if len(self.cache) >= self.cache_size:
            self.write_cache()

    def write_cache(self):
        with self.env.begin(write=True) as txn:
            for key, value in self.cache.items():
                txn.put(key, value)
        self.cache = {}

    def close(self):
        self.cache[b"num-samples"] = str(self.num_samples).encode()
        self.write_cache()

    @staticmethod
    def image_encode(image):
        # image is an Image class instance
        def _encode(image):
            image_byte_arr = six.BytesIO()
            image.save(image_byte_arr, format="JPEG")
            image_enc = image_byte_arr.getvalue()
            return image_enc

        return _encode(image)

    @staticmethod
    def label_encode(label):
        return zlib.compress(json.dumps(label).encode("utf-8"))


class LMDBReader:
    def __init__(self, path, decode=True, mode="PIL"):
        env = lmdb.open(
            path,
            max_readers=100,  # needs to be greater than the total number of workers
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.path = path
        self.txn = env.begin(write=False)
        self.decode = decode
        self.mode = mode
        try:
            self.num_samples = int(self.txn.get("num-samples".encode()))
        except Exception:
            print(path, "has not num-samples key.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # idx starts from 0 to length-1, but lmdb keys start from 1
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
        def _decode(image_enc):
            if mode == "NUMPY":  # BGR
                imgdata = np.frombuffer(image_enc, dtype="uint8")
                image = cv2.imdecode(imgdata, 1)
                return image
            if mode == "PIL":  # RGB
                return Image.open(BytesIO(image_enc))

        try:
            images_enc = pickle.loads(image_enc)
            return [_decode(image_enc) for image_enc in images_enc]
        except pickle.UnpicklingError:
            return _decode(image_enc)

    @staticmethod
    def label_decode(label_enc):
        data = json.loads(zlib.decompress(label_enc).decode("utf-8"))
        return data

