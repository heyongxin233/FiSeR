from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from fiser.data import LMDBReader
from scripts.prepare_lmdb import (
    community_record,
    dataset_records,
    image_long_hash,
    iter_aigibench,
    iter_chameleon,
    iter_genimage,
    iter_wildfake,
    load_expected_sources,
    prepare_lmdb,
    validate_lmdb,
)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _encoded_image(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_folder_adapters_assign_paper_source_names(tmp_path):
    aigibench = tmp_path / "aigibench"
    _image(aigibench / "Model-A" / "Model_A" / "0_real" / "real.png", (1, 2, 3))
    _image(aigibench / "Model-A" / "Model_A" / "1_fake" / "fake.png", (4, 5, 6))
    assert [record.src for record in iter_aigibench(aigibench)] == ["nature", "Model-A"]

    chameleon = tmp_path / "chameleon"
    _image(chameleon / "test" / "0_real" / "real.png", (7, 8, 9))
    _image(chameleon / "test" / "1_fake" / "fake.png", (10, 11, 12))
    assert [record.src for record in iter_chameleon(chameleon)] == ["nature", "AI"]

    genimage = tmp_path / "genimage"
    _image(genimage / "test" / "biggan_imagenet" / "nature" / "real.png", (13, 14, 15))
    _image(genimage / "test" / "biggan_imagenet" / "ai" / "fake.png", (16, 17, 18))
    assert [record.src for record in iter_genimage(genimage)] == ["BigGAN", "nature"]


def test_wildfake_uses_official_split_file_and_path_taxonomy(tmp_path):
    root = tmp_path / "wildfake"
    images = root / "Images"
    real = images / "Real" / "laion" / "real.jpg"
    fake = images / "GAN_based" / "Typical" / "styleGAN" / "fake.png"
    _image(real, (20, 21, 22))
    _image(fake, (23, 24, 25))
    split_file = root / "test.jsonl"
    split_file.write_text(
        json.dumps({"path": "Real/laion/real.jpg"})
        + "\n"
        + json.dumps({"path": "GAN_based/Typical/styleGAN/fake.png"})
        + "\n",
        encoding="utf-8",
    )
    records = list(iter_wildfake(root, "test", split_file))
    assert [record.src for record in records] == ["nature", "styleGAN"]
    assert records[1].metadata["generator_family"] == "GAN_based"


def test_build_is_atomic_and_readable_by_training_loader(tmp_path):
    raw = tmp_path / "raw"
    real_path = raw / "Method-A" / "Method_A" / "0_real" / "real.png"
    fake_path = raw / "Method-A" / "Method_A" / "1_fake" / "fake.png"
    _image(real_path, (30, 31, 32))
    _image(fake_path, (33, 34, 35))
    output = tmp_path / "aigibench.lmdb"
    manifest = tmp_path / "written.jsonl"

    summary = prepare_lmdb(
        iter_aigibench(raw),
        output=output,
        dataset="aigibench",
        split="test",
        workers=2,
        expected_count=2,
        expected_source_count=2,
        expected_sources={"nature", "Method-A"},
        manifest_out=manifest,
        show_progress=False,
    )

    assert summary["samples"] == 2
    assert summary["source_counts"] == {"Method-A": 1, "nature": 1}
    assert (output / "summary.json").is_file()
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 2
    reader = LMDBReader(output)
    try:
        _, real_label = reader[0]
        _, fake_label = reader[1]
    finally:
        reader.close()
    assert real_label["label"] == 1
    assert fake_label["label"] == 0
    assert not Path(real_label["path"]).is_absolute()
    with Image.open(real_path) as image:
        assert real_label["id"] == image_long_hash(image)
    assert validate_lmdb(
        output,
        mode="full",
        expected_count=2,
        expected_source_count=2,
    )["samples"] == 2
    with pytest.raises(ValueError, match="Expected 3 sources, found 2"):
        validate_lmdb(output, mode="keys", expected_source_count=3)


def test_corrupt_input_does_not_leave_partial_lmdb(tmp_path):
    raw = tmp_path / "raw"
    _image(raw / "Method-A" / "Method_A" / "0_real" / "real.png", (40, 41, 42))
    corrupt = raw / "Method-A" / "Method_A" / "1_fake" / "bad.jpg"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not-an-image")
    output = tmp_path / "broken.lmdb"
    with pytest.raises(RuntimeError, match="Failed to process"):
        prepare_lmdb(
            iter_aigibench(raw),
            output=output,
            dataset="aigibench",
            split="test",
            workers=1,
            show_progress=False,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".broken.lmdb.tmp-*"))


def test_community_parquet_adapter_builds_lmdb(tmp_path):
    root = tmp_path / "community"
    root.mkdir()
    table = pa.table(
        {
            "image_data": [_encoded_image((50, 51, 52)), _encoded_image((53, 54, 55))],
            "model_name": ["camera", "Generator-X"],
            "label": [0, 1],
            "split": ["test", "test"],
            "image_name": ["real.png", "fake.png"],
            "subset": ["CompEval", "CompEval"],
        }
    )
    pq.write_table(table, root / "part.parquet")
    output = tmp_path / "community.lmdb"
    summary = prepare_lmdb(
        dataset_records("community", root, "test", None, None),
        output=output,
        dataset="community",
        split="test",
        workers=1,
        expected_count=2,
        expected_sources={"nature", "Generator-X"},
        show_progress=False,
    )
    assert summary["source_counts"] == {"Generator-X": 1, "nature": 1}


def test_community_record_filters_splits():
    row = {
        "image_data": _encoded_image((60, 61, 62)),
        "model_name": "Generator-Y",
        "label": 1,
        "split": "train",
    }
    assert community_record(row, Path("."), "part#0", "test") is None


def test_community_training_records_preserve_generator_classes():
    real = {
        "image_data": _encoded_image((63, 64, 65)),
        "model_name": "LAION",
        "label": 0,
        "split": "train",
    }
    fake = {
        "image_data": _encoded_image((66, 67, 68)),
        "model_name": "organization/generator-model",
        "label": 1,
        "split": "train",
    }
    assert community_record(real, Path("."), "part#0", "train").src == "nature"
    assert (
        community_record(fake, Path("."), "part#1", "train").src
        == "organization/generator-model"
    )


def test_expected_sources_accepts_name_to_index_mapping(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text('{"nature": 0, "Generator-X": 1}\n', encoding="utf-8")
    assert load_expected_sources(path) == {"nature", "Generator-X"}
