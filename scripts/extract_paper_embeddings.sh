#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export TOKENIZERS_PARALLELISM=false

: "${WILDFAKE_TRAIN_LMDB:?Set WILDFAKE_TRAIN_LMDB}"
: "${WILDFAKE_TEST_LMDB:?Set WILDFAKE_TEST_LMDB}"
: "${COMMUNITY_TEST_LMDB:?Set COMMUNITY_TEST_LMDB}"
: "${AIGIBENCH_TEST_LMDB:?Set AIGIBENCH_TEST_LMDB}"
: "${CHAMELEON_TEST_LMDB:?Set CHAMELEON_TEST_LMDB}"
: "${GENIMAGE_TEST_LMDB:?Set GENIMAGE_TEST_LMDB}"

MODEL="${MODEL:-heyongxin233/FiSeR-DINOv3-ViT-L16}"
OUT="${OUT:-outputs/embeddings}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LAYERS=(15 16 17 18 19 20 21 22 23)
mkdir -p "${OUT}"

extract() {
  local lmdb_path="$1"
  local output_path="$2"
  shift 2
  python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
    extract_embeddings.py \
    --model-name "${MODEL}" \
    --dataset-lmdb "${lmdb_path}" \
    --output "${output_path}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --layers "${LAYERS[@]}" \
    "$@"
}

extract "${WILDFAKE_TRAIN_LMDB}" "${OUT}/wildfake_train.pt" \
  --source-map configs/wildfake_sources.json
extract "${WILDFAKE_TEST_LMDB}" "${OUT}/wildfake_test.pt" \
  --source-map configs/wildfake_sources.json
extract "${COMMUNITY_TEST_LMDB}" "${OUT}/community_test.pt" --derive-source-map
extract "${AIGIBENCH_TEST_LMDB}" "${OUT}/aigibench_test.pt" --derive-source-map
extract "${CHAMELEON_TEST_LMDB}" "${OUT}/chameleon_test.pt" --derive-source-map
extract "${GENIMAGE_TEST_LMDB}" "${OUT}/genimage_test.pt" --derive-source-map
