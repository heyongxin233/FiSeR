import json
import zlib

import lmdb
import pytest
from PIL import Image

from fiser.data import build_train_transform, derive_source_map


def _write_label_lmdb(path, sources):
    environment = lmdb.open(str(path), map_size=1 << 20)
    try:
        with environment.begin(write=True) as transaction:
            transaction.put(b"num-samples", str(len(sources)).encode("ascii"))
            for index, source in enumerate(sources, start=1):
                payload = zlib.compress(json.dumps({"src": source, "id": index}).encode("utf-8"))
                transaction.put(b"l-%09d" % index, payload)
    finally:
        environment.close()


def test_derive_source_map_puts_nature_first_and_sorts_sources(tmp_path):
    path = tmp_path / "dataset.lmdb"
    _write_label_lmdb(path, ["sdxl", "nature", "dalle3", "sdxl"])

    assert derive_source_map(path) == {"nature": 0, "dalle3": 1, "sdxl": 2}


def test_derive_source_map_requires_nature_source(tmp_path):
    path = tmp_path / "dataset.lmdb"
    _write_label_lmdb(path, ["sdxl", "dalle3"])

    with pytest.raises(ValueError, match="nature.*absent"):
        derive_source_map(path)


def test_train_transform_keeps_rgb_and_target_size():
    image = Image.new("RGB", (48, 64), color=(120, 80, 40))
    transformed = build_train_transform(32)(image)
    assert transformed.mode == "RGB"
    assert transformed.size == (32, 32)
