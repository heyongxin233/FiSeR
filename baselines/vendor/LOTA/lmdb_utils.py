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
    def __init__(self, path, decode=True, mode='PIL'):
        self.decode = decode
        self.mode = mode

        normalized_path = path.rstrip(os.sep)
        lmdb_kwargs = {
            'max_readers': 100,
            'readonly': True,
            'lock': False,
            'readahead': False,
            'meminit': False,
        }

        if os.path.isfile(normalized_path):
            lmdb_kwargs['subdir'] = False
        elif not os.path.isdir(normalized_path) or not os.path.isfile(os.path.join(normalized_path, 'data.mdb')):
            raise FileNotFoundError(
                f'Provided LMDB path {path} is neither a data.mdb file nor a directory containing data.mdb.'
            )

        env = lmdb.open(normalized_path, **lmdb_kwargs)
        self.txn = env.begin(write=False)
        num_samples = self.txn.get(b'num-samples')
        if num_samples is None:
            raise ValueError(f'LMDB at {path} is missing the num-samples key.')
        self.num_samples = int(num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx += 1
        image_key = b'i-%09d' % idx
        label_key = b'l-%09d' % idx
        image_enc = self.txn.get(image_key)
        label_enc = self.txn.get(label_key)
        if image_enc is None or label_enc is None:
            raise IndexError(f'Failed to fetch LMDB sample at index {idx - 1}.')
        if self.decode:
            return self.image_decode(image_enc, self.mode), self.label_decode(label_enc)
        return image_enc, label_enc

    def get_label(self, idx):
        return self.label_decode(self.txn.get(b'l-%09d' % (idx + 1)))

    @staticmethod
    def image_decode(image_enc, mode='PIL'):
        def _decode(encoded_image):
            if mode == 'NUMPY':
                imgdata = np.frombuffer(encoded_image, dtype='uint8')
                return cv2.imdecode(imgdata, 1)
            if mode == 'PIL':
                return Image.open(BytesIO(encoded_image))
            raise ValueError(f'Unsupported decode mode: {mode}')

        try:
            images_enc = pickle.loads(image_enc)
            return [_decode(encoded_image) for encoded_image in images_enc]
        except pickle.UnpicklingError:
            return _decode(image_enc)

    @staticmethod
    def label_decode(label_enc):
        return json.loads(zlib.decompress(label_enc).decode('utf-8'))
