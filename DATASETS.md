# Dataset Preparation

FiSeR does not redistribute images or generated LMDB files. Download each
dataset from its official release, then run `scripts/prepare_lmdb.py` locally.
The command fails on unreadable images, writes through a temporary directory,
and only exposes the final LMDB after full validation succeeds.

## Reference Data

The LMDBs used by the released pipeline contain the following samples. Pass
`--expected-count` to detect a different dataset revision or incomplete
download instead of silently evaluating it.

| Split | Samples | Sources |
| --- | ---: | ---: |
| WildFake train | 2,643,891 | 19 |
| WildFake test | 660,978 | 19 |
| Community Forensics train | 2,971,612 | 4,783 |
| Community Forensics test | 51,220 | 22 |
| AIGIBench test | 212,802 | 26 |
| Chameleon test | 26,033 | 2 |
| GenImage test | 100,000 | 9 |

Dataset references are WildFake (arXiv:2402.11843), Community Forensics
(CVPR 2025), AIGIBench (arXiv:2505.12335), Chameleon from the AIDE evaluation
suite (arXiv:2406.19435), and GenImage (NeurIPS 2023). Access and licensing
remain governed by the original releases.

## Installation

The normal project installation includes Pillow, LMDB, and PyArrow, which is
needed for the Community Forensics parquet release:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## WildFake

WildFake uses a shared `Images/` tree, so exact train/test membership must come
from the official split annotation. FiSeR intentionally does not invent a new
random split. The split file may be JSONL, JSON, CSV, or plain text and must
contain image paths; paths can be absolute or relative to `Images/`.

```text
WildFake/
├── Images/
│   ├── Real/
│   ├── GAN_based/
│   ├── Diffusion_based/
│   └── Other_based/
├── train.jsonl
└── test.jsonl
```

```bash
python scripts/prepare_lmdb.py \
  --dataset wildfake --root /path/to/WildFake \
  --split train --split-file /path/to/WildFake/train.jsonl \
  --output /path/to/processed/wildfake_train.lmdb \
  --expected-count 2643891 \
  --expected-sources configs/wildfake_sources.json

python scripts/prepare_lmdb.py \
  --dataset wildfake --root /path/to/WildFake \
  --split test --split-file /path/to/WildFake/test.jsonl \
  --output /path/to/processed/wildfake_test.lmdb \
  --expected-count 660978 \
  --expected-sources configs/wildfake_sources.json
```

One JSONL split entry can be as small as:

```json
{"path":"GAN_based/Typical/styleGAN/example.png"}
```

## Community Forensics

The full training release is `OwensLab/CommunityForensics`, whose Hugging Face
split is named `Systematic+Manual`. After downloading the parquet snapshot,
point `--root` at the directory containing its `manual/` and `systematic/`
shards. The adapter recursively reads both trees and selects rows whose
internal `split` field is `train`.

```bash
hf download OwensLab/CommunityForensics --repo-type dataset \
  --local-dir /path/to/CommunityForensics
```

```text
CommunityForensics/
└── data/
    ├── manual/
    │   └── *.parquet
    └── systematic/
        └── *.parquet
```

```bash
python scripts/prepare_lmdb.py \
  --dataset community --root /path/to/CommunityForensics \
  --split train --output /path/to/processed/community_train.lmdb \
  --expected-count 2971612 --expected-source-count 4783
```

This command targets the full release used by the local experiments, not
`CommunityForensics-Small`. The full LMDB has one `nature` source and 4,782
synthetic generator sources. Official labels use `real=0, fake=1`; the
converter preserves every synthetic `model_name` as `src` and normalizes real
rows to `src=nature`.

For evaluation, point `--root` at the downloaded
`OwensLab/CommunityForensics-Eval` release. The same adapter supports both
`image_data` and Hugging Face `image={bytes,path}` parquet columns.

```bash
python scripts/prepare_lmdb.py \
  --dataset community --root /path/to/CommunityForensics-Eval \
  --split test --output /path/to/processed/community_test.lmdb \
  --expected-count 51220 \
  --expected-sources configs/datasets/community_sources.json
```

## AIGIBench

The adapter expects each method directory to contain natural and synthetic
subdirectories. The intermediate directory name may differ from the method
name; the outer method name becomes `src`.

```text
AIGIBench/test/
└── METHOD/
    └── METHOD_VARIANT/
        ├── 0_real/
        └── 1_fake/
```

```bash
python scripts/prepare_lmdb.py \
  --dataset aigibench --root /path/to/AIGIBench/test \
  --output /path/to/processed/aigibench_test.lmdb \
  --expected-count 212802 \
  --expected-sources configs/datasets/aigibench_sources.json
```

## Chameleon

```text
Chameleon/
└── test/
    ├── 0_real/
    └── 1_fake/
```

```bash
python scripts/prepare_lmdb.py \
  --dataset chameleon --root /path/to/Chameleon \
  --output /path/to/processed/chameleon_test.lmdb \
  --expected-count 26033 \
  --expected-sources configs/datasets/chameleon_sources.json
```

## GenImage

```text
genimage_test/test/
├── adm_imagenet/{nature,ai}/
├── biggan_imagenet/{nature,ai}/
├── glide_imagenet/{nature,ai}/
├── midjourney_imagenet/{nature,ai}/
├── sdv4_imagenet/{nature,ai}/
├── sdv5_imagenet/{nature,ai}/
├── vqdm_imagenet/{nature,ai}/
└── wukong_imagenet/{nature,ai}/
```

```bash
python scripts/prepare_lmdb.py \
  --dataset genimage --root /path/to/GenImage/genimage_test/test \
  --output /path/to/processed/genimage_test.lmdb \
  --expected-count 100000 \
  --expected-sources configs/datasets/genimage_sources.json
```

## Custom Layouts

For a different release layout, provide a JSONL manifest instead of modifying
the converter. Paths may be relative to `--root`:

```json
{"path":"real/0001.jpg","src":"nature"}
{"path":"fake/0001.png","src":"my_generator"}
```

```bash
python scripts/prepare_lmdb.py \
  --dataset manifest --root /path/to/images --manifest split.jsonl \
  --output /path/to/processed/custom.lmdb \
  --manifest-out /path/to/processed/custom.written.jsonl
```

## Data Contract

Images are decoded as RGB and, by default, encoded as JPEG quality 75 to match
the legacy LMDB generation pipeline used by the experiments. Use
`--image-encoding original` only for a new protocol; it may change metrics.
Each label is zlib-compressed JSON with these required fields:

```json
{"path":"relative/path.jpg","src":"nature","label":1,"id":123}
```

The stored legacy label is `natural=1, synthetic=0`. FiSeR never trusts that
field for its public prediction convention: the training/evaluation loader
derives `natural=0, synthetic=1` from `src`.

Every sample ID is a deterministic SHA-256-derived hash of decoded RGB pixels.
The converter writes in batches, preserves deterministic traversal order,
records per-source counts in `summary.json`, and performs a full second-pass
image/key/label validation before atomically publishing the LMDB directory.
Validation can be rerun independently:

```bash
python scripts/check_dataset.py /path/to/processed/aigibench_test.lmdb \
  --expected-count 212802
```

For the full Community Forensics training set, rerun the cheaper key/label
check without decoding 2.97 million images:

```bash
python scripts/check_dataset.py /path/to/processed/community_train.lmdb \
  --mode keys --expected-count 2971612 --expected-source-count 4783
```
