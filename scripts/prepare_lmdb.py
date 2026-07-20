from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import lmdb
from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NATURE_SOURCE = "nature"
GENIMAGE_SOURCES = {
    "adm": "ADM",
    "biggan": "BigGAN",
    "glide": "glide",
    "midjourney": "Midjourney",
    "sdv4": "stable_diffusion_v_1_4",
    "sdv5": "stable_diffusion_v_1_5",
    "vqdm": "VQDM",
    "wukong": "wukong",
}


@dataclass(frozen=True)
class SampleRecord:
    src: str
    origin: str
    path: str | None = None
    image_bytes: bytes | None = None
    metadata: dict[str, Any] | None = None


def encode_label(label: dict[str, Any]) -> bytes:
    rendered = json.dumps(label, ensure_ascii=False, default=str, separators=(",", ":"))
    return zlib.compress(rendered.encode("utf-8"))


def image_long_hash(image: Image.Image) -> int:
    digest = hashlib.sha256(image.convert("RGB").tobytes()).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _sorted_images(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def _origin(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _record(path: Path, root: Path, src: str, dataset: str, split: str) -> SampleRecord:
    return SampleRecord(
        path=str(path),
        origin=_origin(path, root),
        src=src,
        metadata={"dataset": dataset, "split": split},
    )


def _label_directory(parts: tuple[str, ...]) -> tuple[int, str] | None:
    matches = [(index, part) for index, part in enumerate(parts) if part in {"0_real", "1_fake"}]
    return matches[-1] if matches else None


def iter_aigibench(root: Path, split: str = "test") -> Iterator[SampleRecord]:
    found = 0
    for path in _sorted_images(root):
        relative = path.relative_to(root)
        marker = _label_directory(relative.parts)
        if marker is None:
            continue
        _, label_dir = marker
        src = NATURE_SOURCE if label_dir == "0_real" else relative.parts[0]
        found += 1
        yield _record(path, root, src, "aigibench", split)
    if found == 0:
        raise ValueError("AIGIBench needs METHOD/.../{0_real,1_fake}/IMAGE under --root")


def iter_chameleon(root: Path, split: str = "test") -> Iterator[SampleRecord]:
    found = 0
    for path in _sorted_images(root):
        relative = path.relative_to(root)
        marker = _label_directory(relative.parts)
        if marker is None:
            continue
        _, label_dir = marker
        found += 1
        yield _record(path, root, NATURE_SOURCE if label_dir == "0_real" else "AI", "chameleon", split)
    if found == 0:
        raise ValueError("Chameleon needs test/{0_real,1_fake}/IMAGE under --root")


def iter_genimage(root: Path, split: str = "test") -> Iterator[SampleRecord]:
    found = 0
    for path in _sorted_images(root):
        relative = path.relative_to(root)
        marker_indices = [
            index for index, part in enumerate(relative.parts) if part.lower() in {"nature", "ai"}
        ]
        if not marker_indices:
            continue
        marker_index = marker_indices[-1]
        if marker_index == 0:
            continue
        label_dir = relative.parts[marker_index].lower()
        generator_dir = relative.parts[marker_index - 1].lower().removesuffix("_imagenet")
        if generator_dir not in GENIMAGE_SOURCES:
            raise ValueError(f"Unsupported GenImage generator directory: {relative.parts[marker_index - 1]}")
        src = NATURE_SOURCE if label_dir == "nature" else GENIMAGE_SOURCES[generator_dir]
        found += 1
        yield _record(path, root, src, "genimage", split)
    if found == 0:
        raise ValueError("GenImage needs GENERATOR_imagenet/{nature,ai}/IMAGE under --root")


def _wildfake_source(path: Path, images_root: Path) -> tuple[str, str]:
    parts = path.relative_to(images_root).parts
    categories = {"real", "gan_based", "diffusion_based", "other_based"}
    try:
        category_index = next(
            index for index, part in enumerate(parts) if part.lower() in categories
        )
    except StopIteration as exc:
        raise ValueError(f"Cannot infer a WildFake source from {path}") from exc
    category = parts[category_index].lower()
    if category == "real":
        return NATURE_SOURCE, "Real"
    offset = category_index + 1
    if category in {"gan_based", "other_based"} and parts[offset].lower() in {
        "advanced",
        "typical",
    }:
        offset += 1
    if offset >= len(parts) - 1:
        raise ValueError(f"Cannot infer a WildFake generator from {path}")
    return parts[offset], parts[category_index]


def _split_entries(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("A JSON split file must contain a list")
        for item in payload:
            yield item if isinstance(item, dict) else {"path": item}
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if suffix == ".jsonl":
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Split line {line_number} must be an object")
                yield item
            else:
                yield {"path": line}


def _entry_path(entry: dict[str, Any]) -> str:
    for key in ("path", "image_path", "file", "filename"):
        if entry.get(key):
            return str(entry[key])
    raise ValueError(f"Split entry has no image path: {entry}")


def _resolve_split_path(raw: str, images_root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = images_root / path
    if candidate.is_file():
        return candidate.resolve()
    normalized = raw.replace("\\", "/")
    marker = "/Images/"
    if marker in normalized:
        candidate = images_root / normalized.split(marker, 1)[1]
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Split entry does not resolve under {images_root}: {raw}")


def iter_wildfake(
    root: Path,
    split: str,
    split_file: Path | None = None,
) -> Iterator[SampleRecord]:
    images_root = root / "Images" if (root / "Images").is_dir() else root
    if split_file is None:
        split_root = images_root / split
        if split in {"train", "test"} and split_root.is_dir():
            images_root = split_root
        elif split in {"train", "test"} and images_root.name.lower() != split:
            raise ValueError(
                "WildFake train/test requires the official --split-file unless --root already "
                "points to a split directory"
            )
        paths: Iterable[tuple[Path, dict[str, Any]]] = (
            (path, {}) for path in _sorted_images(images_root)
        )
    else:
        paths = (
            (_resolve_split_path(_entry_path(entry), images_root), entry)
            for entry in _split_entries(split_file)
        )

    found = 0
    for path, entry in paths:
        inferred_source, generator_family = _wildfake_source(path, images_root)
        src = str(entry.get("src") or inferred_source)
        if entry.get("src") and src != inferred_source:
            raise ValueError(
                f"WildFake split source {src!r} disagrees with path source "
                f"{inferred_source!r}: {path}"
            )
        metadata = {
            "dataset": "wildfake",
            "split": split,
            "generator_family": generator_family,
        }
        found += 1
        yield SampleRecord(
            path=str(path),
            origin=_origin(path, images_root),
            src=src,
            metadata=metadata,
        )
    if found == 0:
        raise ValueError("No WildFake images were found")


def iter_manifest(path: Path, root: Path | None = None) -> Iterator[SampleRecord]:
    base = (root or path.parent).resolve()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("path") or not row.get("src"):
                raise ValueError(f"Manifest line {line_number} needs path and src")
            image_path = Path(str(row["path"])).expanduser()
            if not image_path.is_absolute():
                image_path = base / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest line {line_number}: {image_path}")
            metadata = {
                str(key): value
                for key, value in row.items()
                if key not in {"path", "src", "label", "id"}
            }
            yield SampleRecord(
                path=str(image_path.resolve()),
                origin=_origin(image_path, base),
                src=str(row["src"]),
                metadata=metadata,
            )


def _community_image(row: dict[str, Any], root: Path) -> tuple[str | None, bytes | None]:
    value = row.get("image_data", row.get("image"))
    if isinstance(value, memoryview):
        return None, value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return None, bytes(value)
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, (bytes, bytearray)):
            return None, bytes(raw)
        value = value.get("path")
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return str(path.resolve()), None
    raise ValueError("Community Forensics row has no image bytes/path")


def community_record(
    row: dict[str, Any],
    root: Path,
    origin: str,
    requested_split: str | None,
) -> SampleRecord | None:
    row_split = str(row.get("split") or "")
    if requested_split and row_split and row_split != requested_split:
        return None
    raw_src = str(row.get("model_name", row.get("src", "")))
    raw_label = row.get("label")
    natural = raw_src.lower() in {"human", "nature", "real"}
    if not natural and raw_label is not None:
        normalized = str(raw_label).strip().lower()
        natural = normalized in {"0", "false", "human", "nature", "real"}
    src = NATURE_SOURCE if natural else raw_src
    if not src:
        raise ValueError(f"Community Forensics row has no model_name/src: {origin}")
    image_path, image_bytes = _community_image(row, root)
    metadata = {
        key: row.get(key)
        for key in ("image_name", "prompt", "real_source", "subset")
        if key in row
    }
    metadata.update({"dataset": "community", "split": row_split or requested_split or "all"})
    return SampleRecord(
        path=image_path,
        image_bytes=image_bytes,
        origin=origin,
        src=src,
        metadata=metadata,
    )


def iter_community(root: Path, split: str | None = None) -> Iterator[SampleRecord]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - exercised by installation guard
        raise RuntimeError("Community Forensics conversion requires pyarrow") from exc
    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    found = 0
    for parquet_path in parquet_files:
        row_index = 0
        table = parquet.ParquetFile(parquet_path)
        for batch in table.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                origin = f"{_origin(parquet_path, root)}#{row_index}"
                row_index += 1
                record = community_record(row, root, origin, split)
                if record is not None:
                    found += 1
                    yield record
    if found == 0:
        raise ValueError(f"No Community Forensics rows matched split={split!r}")


def dataset_records(
    dataset: str,
    root: Path | None,
    split: str | None,
    split_file: Path | None,
    manifest: Path | None,
) -> Iterator[SampleRecord]:
    if dataset == "manifest":
        if manifest is None:
            raise ValueError("--manifest is required for dataset=manifest")
        yield from iter_manifest(manifest, root)
        return
    if root is None:
        raise ValueError(f"--root is required for dataset={dataset}")
    root = root.resolve()
    if dataset == "wildfake":
        if not split:
            raise ValueError("WildFake requires --split train, test, or all")
        yield from iter_wildfake(root, split, split_file)
    elif dataset == "community":
        yield from iter_community(root, split)
    elif dataset == "aigibench":
        yield from iter_aigibench(root, split or "test")
    elif dataset == "chameleon":
        yield from iter_chameleon(root, split or "test")
    elif dataset == "genimage":
        yield from iter_genimage(root, split or "test")
    else:  # pragma: no cover - argparse guards this path
        raise ValueError(f"Unsupported dataset: {dataset}")


def _encode_record(task: tuple[SampleRecord, str, int]) -> tuple[bytes, dict[str, Any]]:
    record, image_encoding, jpeg_quality = task
    try:
        raw = record.image_bytes
        if raw is None:
            if record.path is None:
                raise ValueError("record has neither path nor image bytes")
            raw = Path(record.path).read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            rgb = image.convert("RGB")
        sample_id = image_long_hash(rgb)
        if image_encoding == "jpeg":
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=jpeg_quality)
            encoded = buffer.getvalue()
        else:
            encoded = raw
        label = dict(record.metadata or {})
        label.update(
            {
                "path": record.origin,
                "src": record.src,
                "label": int(record.src == NATURE_SOURCE),
                "id": sample_id,
            }
        )
        return encoded, label
    except Exception as exc:
        raise RuntimeError(f"Failed to process {record.origin}: {exc}") from exc


class _BatchWriter:
    def __init__(self, path: Path, map_size: int, commit_every: int) -> None:
        path.mkdir(parents=True)
        self.env = lmdb.open(str(path), map_size=map_size, subdir=True)
        self.transaction = self.env.begin(write=True)
        self.commit_every = max(1, int(commit_every))
        self.count = 0
        self.closed = False

    def add(self, image: bytes, label: dict[str, Any]) -> None:
        self.count += 1
        self.transaction.put(f"i-{self.count:09d}".encode(), image)
        self.transaction.put(f"l-{self.count:09d}".encode(), encode_label(label))
        if self.count % self.commit_every == 0:
            self.transaction.commit()
            self.transaction = self.env.begin(write=True)

    def finish(self) -> None:
        self.transaction.put(b"num-samples", str(self.count).encode())
        self.transaction.commit()
        self.env.sync()
        self.env.close()
        self.closed = True

    def abort(self) -> None:
        if self.closed:
            return
        self.transaction.abort()
        self.env.close()
        self.closed = True


def validate_lmdb(
    path: str | Path,
    mode: str = "full",
    expected_count: int | None = None,
    expected_source_count: int | None = None,
    expected_sources: set[str] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    source_counts: Counter[str] = Counter()
    ids: set[int] = set()
    duplicate_ids = 0
    try:
        with env.begin(write=False) as transaction:
            raw_count = transaction.get(b"num-samples")
            if raw_count is None:
                raise ValueError("LMDB is missing num-samples")
            count = int(raw_count)
            if expected_count is not None and count != int(expected_count):
                raise ValueError(f"Expected {expected_count} samples, found {count}")
            expected_entries = 2 * count + 1
            actual_entries = int(transaction.stat()["entries"])
            if actual_entries != expected_entries:
                raise ValueError(
                    f"LMDB has {actual_entries} entries; expected {expected_entries} for {count} samples"
                )

            image_count = 0
            label_count = 0
            key_cursor = transaction.cursor()
            for key in key_cursor.iternext(keys=True, values=False):
                if key.startswith(b"i-"):
                    image_count += 1
                    if key != f"i-{image_count:09d}".encode():
                        raise ValueError(f"Unexpected image key: {key!r}")
                elif key.startswith(b"l-"):
                    label_count += 1
                    if key != f"l-{label_count:09d}".encode():
                        raise ValueError(f"Unexpected label key: {key!r}")
                elif key != b"num-samples":
                    raise ValueError(f"Unexpected LMDB key: {key!r}")
            if image_count != count or label_count != count:
                raise ValueError(
                    f"Expected {count} image/label keys, found images={image_count}, "
                    f"labels={label_count}"
                )

            label_cursor = transaction.cursor()
            if count and not label_cursor.set_range(b"l-000000001"):
                raise ValueError("LMDB has no label keys")
            for index in range(1, count + 1):
                key, label_bytes = label_cursor.item()
                if key != f"l-{index:09d}".encode():
                    raise ValueError(f"Unexpected label key while scanning: {key!r}")
                label = json.loads(zlib.decompress(label_bytes).decode("utf-8"))
                src = str(label.get("src", ""))
                if not src:
                    raise ValueError(f"Sample {index} has no src")
                if int(label.get("label", -1)) != int(src == NATURE_SOURCE):
                    raise ValueError(f"Sample {index} has an inconsistent stored label")
                sample_id = int(label["id"])
                if sample_id in ids:
                    duplicate_ids += 1
                ids.add(sample_id)
                source_counts[src] += 1
                if index < count:
                    label_cursor.next()

            if mode == "full":
                image_cursor = transaction.cursor()
                if count and not image_cursor.set_range(b"i-000000001"):
                    raise ValueError("LMDB has no image keys")
                for index in range(1, count + 1):
                    key, image_bytes = image_cursor.item()
                    if key != f"i-{index:09d}".encode():
                        raise ValueError(f"Unexpected image key while scanning: {key!r}")
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.verify()
                    if index < count:
                        image_cursor.next()
            if NATURE_SOURCE not in source_counts or len(source_counts) < 2:
                raise ValueError("LMDB must contain natural and synthetic samples")
            if expected_source_count is not None and len(source_counts) != expected_source_count:
                raise ValueError(
                    f"Expected {expected_source_count} sources, found {len(source_counts)}"
                )
            if expected_sources is not None and set(source_counts) != expected_sources:
                missing = sorted(expected_sources - set(source_counts))
                unexpected = sorted(set(source_counts) - expected_sources)
                raise ValueError(f"Source mismatch: missing={missing}, unexpected={unexpected}")
    finally:
        env.close()
    return {
        "samples": count,
        "source_counts": dict(sorted(source_counts.items())),
        "duplicate_ids": duplicate_ids,
        "validation": mode,
    }


def load_expected_sources(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key) for key in payload}
    if isinstance(payload, list):
        return {str(value) for value in payload}
    raise ValueError("Expected sources must be a JSON object or list")


def prepare_lmdb(
    records: Iterable[SampleRecord],
    output: Path,
    dataset: str,
    split: str | None,
    workers: int = 1,
    image_encoding: str = "jpeg",
    jpeg_quality: int = 75,
    map_size: int = 1 << 40,
    commit_every: int = 1000,
    validation: str = "full",
    expected_count: int | None = None,
    expected_source_count: int | None = None,
    expected_sources: set[str] | None = None,
    manifest_out: Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    manifest_tmp = None
    manifest_handle = None
    if manifest_out is not None:
        manifest_out = manifest_out.resolve()
        if manifest_out.exists():
            raise FileExistsError(f"Refusing to overwrite manifest: {manifest_out}")
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_tmp = manifest_out.with_name(f".{manifest_out.name}.tmp-{os.getpid()}")
        manifest_handle = manifest_tmp.open("w", encoding="utf-8")

    writer: _BatchWriter | None = None
    pool: mp.pool.Pool | None = None
    try:
        writer = _BatchWriter(temporary, map_size=map_size, commit_every=commit_every)
        tasks = ((record, image_encoding, jpeg_quality) for record in records)
        if workers > 1:
            pool = mp.Pool(processes=workers)
            encoded_records = pool.imap(_encode_record, tasks, chunksize=8)
        else:
            encoded_records = map(_encode_record, tasks)
        for image_bytes, label in tqdm(
            encoded_records,
            desc="Writing LMDB",
            unit="image",
            disable=not show_progress,
        ):
            writer.add(image_bytes, label)
            if manifest_handle is not None:
                manifest_handle.write(json.dumps(label, ensure_ascii=False, default=str) + "\n")
        if pool is not None:
            pool.close()
            pool.join()
            pool = None
        if writer.count == 0:
            raise ValueError("No samples were discovered")
        if expected_count is not None and writer.count != int(expected_count):
            raise ValueError(f"Expected {expected_count} samples, discovered {writer.count}")
        writer.finish()
        writer = None
        validation_result = validate_lmdb(
            temporary,
            mode=validation,
            expected_count=expected_count,
            expected_source_count=expected_source_count,
            expected_sources=expected_sources,
        )
        summary = {
            "dataset": dataset,
            "split": split or "all",
            "image_encoding": image_encoding,
            "jpeg_quality": jpeg_quality if image_encoding == "jpeg" else None,
            **validation_result,
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        if manifest_handle is not None:
            manifest_handle.close()
            manifest_handle = None
            assert manifest_tmp is not None and manifest_out is not None
            os.replace(manifest_tmp, manifest_out)
        return summary
    except Exception:
        if pool is not None:
            pool.terminate()
            pool.join()
        if writer is not None:
            writer.abort()
        if manifest_handle is not None:
            manifest_handle.close()
        if temporary.exists():
            shutil.rmtree(temporary)
        if manifest_tmp is not None and manifest_tmp.exists():
            manifest_tmp.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate a FiSeR-compatible LMDB")
    parser.add_argument(
        "--dataset",
        choices=["wildfake", "community", "aigibench", "chameleon", "genimage", "manifest"],
    )
    parser.add_argument("--root", type=Path, help="Raw dataset root")
    parser.add_argument("--manifest", type=Path, help="JSONL path for dataset=manifest")
    parser.add_argument("--split", help="Dataset split, such as train or test")
    parser.add_argument("--split-file", type=Path, help="Official WildFake split list")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, help="Optional manifest of written samples")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--image-encoding", choices=["jpeg", "original"], default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--map-size", type=int, default=1 << 40)
    parser.add_argument("--commit-every", type=int, default=1000)
    parser.add_argument("--validation", choices=["full", "keys"], default="full")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--expected-sources", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset or ("manifest" if args.manifest else None)
    if dataset is None:
        raise ValueError("Set --dataset or provide --manifest")
    records = dataset_records(dataset, args.root, args.split, args.split_file, args.manifest)
    summary = prepare_lmdb(
        records,
        output=args.output,
        dataset=dataset,
        split=args.split,
        workers=max(1, args.workers),
        image_encoding=args.image_encoding,
        jpeg_quality=args.jpeg_quality,
        map_size=args.map_size,
        commit_every=args.commit_every,
        validation=args.validation,
        expected_count=args.expected_count,
        expected_source_count=args.expected_source_count,
        expected_sources=load_expected_sources(args.expected_sources),
        manifest_out=args.manifest_out,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
