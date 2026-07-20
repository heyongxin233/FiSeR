#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export TOKENIZERS_PARALLELISM=false

TRAIN_LMDB="${TRAIN_LMDB:-/path/to/Image_detection_datasets/WildFake/train}"
OUT="${OUT:-outputs/reproduction_training}"

python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-2}" train.py \
  --config configs/fiser_wildfake.yaml \
  --override train_lmdb="${TRAIN_LMDB}" \
  --override output_dir="${OUT}"
