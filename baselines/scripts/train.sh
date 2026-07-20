#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD="${1:?Usage: $0 METHOD}"
shift

: "${TRAIN_LMDB:?Set TRAIN_LMDB to a prepared training LMDB}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "CUDA is required")'

NPROC="${NPROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/baselines/${METHOD}}"
RUN_NAME="${RUN_NAME:-${METHOD}_TrainingDataset}"

VAL_ARGS=()
EVAL_DATA_ARGS=()
if [[ -n "${VAL_LMDB:-}" ]]; then
  VAL_ARGS=(--val_lmdb "${VAL_LMDB}")
  EVAL_DATA_ARGS=(--eval_data_path "${VAL_LMDB}")
fi

case "${METHOD}" in
  resnet50)
    cd "${ROOT}"
    exec torchrun --standalone --nproc_per_node="${NPROC}" -m baselines.train \
      --config baselines/configs/resnet50_wildfake.yaml \
      --train-lmdb "${TRAIN_LMDB}" --output-dir "${OUTPUT_DIR}" \
      --epochs "${EPOCHS}" "$@"
    ;;
  clip-detection)
    cd "${ROOT}"
    exec torchrun --standalone --nproc_per_node="${NPROC}" -m baselines.train \
      --config baselines/configs/clip_detection_wildfake.yaml \
      --train-lmdb "${TRAIN_LMDB}" --output-dir "${OUTPUT_DIR}" \
      --epochs "${EPOCHS}" "$@"
    ;;
  cnn-detection)
    cd "${ROOT}/baselines/vendor/CNNDetection-master"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --name "${RUN_NAME}" --train_lmdb "${TRAIN_LMDB}" \
      --checkpoints_dir "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" --num_threads "${NUM_WORKERS}" \
      "${VAL_ARGS[@]}" "$@"
    ;;
  lgrad)
    cd "${ROOT}/baselines/vendor/LGrad-master/CNNDetection"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --distributed --name "${RUN_NAME}" --train_lmdb "${TRAIN_LMDB}" \
      --checkpoints_dir "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" --num_threads "${NUM_WORKERS}" \
      "${VAL_ARGS[@]}" "$@"
    ;;
  gram-net)
    cd "${ROOT}/baselines/vendor/Gram-Net-main"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --distributed --name "${RUN_NAME}" --train_lmdb "${TRAIN_LMDB}" \
      --checkpoints_dir "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" --num_threads "${NUM_WORKERS}" \
      "${VAL_ARGS[@]}" "$@"
    ;;
  freqnet)
    cd "${ROOT}/baselines/vendor/FreqNet-DeepfakeDetection-main"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --name "${RUN_NAME}" --train_lmdb "${TRAIN_LMDB}" \
      --checkpoints_dir "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --niter "${EPOCHS}" --num_threads "${NUM_WORKERS}" \
      "${VAL_ARGS[@]}" "$@"
    ;;
  npr)
    cd "${ROOT}/baselines/vendor/NPR-DeepfakeDetection-main"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --distributed --name "${RUN_NAME}" --train_lmdb "${TRAIN_LMDB}" \
      --checkpoints_dir "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --niter "${EPOCHS}" --num_threads "${NUM_WORKERS}" \
      "${VAL_ARGS[@]}" "$@"
    ;;
  safe)
    cd "${ROOT}/baselines/vendor/SAFE-main"
    exec torchrun --standalone --nproc_per_node="${NPROC}" main_finetune.py \
      --distributed --model SAFE --data_path "${TRAIN_LMDB}" \
      --output_dir "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" --num_workers "${NUM_WORKERS}" \
      --blr "${LR:-0.0001}" "${EVAL_DATA_ARGS[@]}" "$@"
    ;;
  lasted)
    cd "${ROOT}/baselines/vendor/LASTED"
    exec torchrun --standalone --nproc_per_node="${NPROC}" main.py \
      --distributed --isTrain 1 --train_lmdb "${TRAIN_LMDB}" \
      "${VAL_ARGS[@]}" --weights "${OUTPUT_DIR}" \
      --batch_size "${BATCH_SIZE}" --epoches "${EPOCHS}" "$@"
    ;;
  unifd)
    cd "${ROOT}/baselines/vendor/UniversalFakeDetect"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --distributed --name "${RUN_NAME}" --train_lmdb "${TRAIN_LMDB}" \
      --checkpoints_dir "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" --num_threads "${NUM_WORKERS}" \
      "${VAL_ARGS[@]}" "$@"
    ;;
  aide)
    AIDE_INIT_ARGS=()
    if [[ -n "${RESNET_PATH:-}" ]]; then
      AIDE_INIT_ARGS+=(--resnet_path "${RESNET_PATH}")
    fi
    if [[ -n "${CONVNEXT_PATH:-}" ]]; then
      AIDE_INIT_ARGS+=(--convnext_path "${CONVNEXT_PATH}")
    fi
    cd "${ROOT}/baselines/vendor/AIDE-main"
    exec torchrun --standalone --nproc_per_node="${NPROC}" main_finetune.py \
      --model AIDE --data_path "${TRAIN_LMDB}" --output_dir "${OUTPUT_DIR}" \
      --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" \
      --num_workers "${NUM_WORKERS}" --blr "${LR:-0.0001}" \
      "${EVAL_DATA_ARGS[@]}" "${AIDE_INIT_ARGS[@]}" "$@"
    ;;
  dire)
    DIRE_VAL_ARGS=()
    if [[ -n "${VAL_LMDB:-}" ]]; then
      DIRE_VAL_ARGS=(val_lmdb "${VAL_LMDB}")
    fi
    cd "${ROOT}/baselines/vendor/DIRE"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --exp_name "${RUN_NAME}" train_lmdb "${TRAIN_LMDB}" \
      exp_root "${OUTPUT_DIR}" nepoch "${EPOCHS}" \
      batch_size "${BATCH_SIZE}" num_workers "${NUM_WORKERS}" \
      "${DIRE_VAL_ARGS[@]}" "$@"
    ;;
  effort)
    cd "${ROOT}/baselines/vendor/Effort-AIGI-Detection/UniversalFakeDetect_Benchmark"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --distributed --compile --use_svd --arch CLIP:ViT-L/14_svd \
      --name "${RUN_NAME}" --train_lmdb "${TRAIN_LMDB}" \
      --checkpoints_dir "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --niter "${EPOCHS}" --num_threads "${NUM_WORKERS}" \
      --lr "${LR:-0.0004}" --loss_freq 0 "${VAL_ARGS[@]}" "$@"
    ;;
  spai)
    : "${PRETRAINED:?Set PRETRAINED to the MFM ViT-B/16 initialization checkpoint}"
    cd "${ROOT}/baselines/vendor/spai"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --train_lmdb "${TRAIN_LMDB}" --output "${OUTPUT_DIR}" \
      --pretrained "${PRETRAINED}" --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" --num_workers "${NUM_WORKERS}" \
      --precision "${PRECISION:-bf16}" "$@"
    ;;
  c2p-clip)
    cd "${ROOT}/baselines/vendor/C2P-CLIP-DeepfakeDetection"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --distributed --amp --amp_dtype "${AMP_DTYPE:-bf16}" \
      --name "${RUN_NAME}" --train_lmdb "${TRAIN_LMDB}" \
      --checkpoints_dir "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --niter "${EPOCHS}" --num_threads "${NUM_WORKERS}" \
      --clip "${CLIP_MODEL:-openai/clip-vit-large-patch14}" \
      --lr "${LR:-0.0002}" --claloss 8.0 \
      --lora_r 6 --lora_alpha 6 --lora_dropout 0.5 \
      "${VAL_ARGS[@]}" "$@"
    ;;
  lota)
    cd "${ROOT}/baselines/vendor/LOTA"
    exec torchrun --standalone --nproc_per_node="${NPROC}" train.py \
      --distributed --name "${RUN_NAME}" --train_lmdb "${TRAIN_LMDB}" \
      --save_path "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" --num_workers "${NUM_WORKERS}" \
      --precision "${PRECISION:-bf16}" --scheduler cosine --warmup_epochs 1 \
      --min_lr 0 --bit_mode scaling --patch_size 32 --patch_mode random \
      "${VAL_ARGS[@]}" "$@"
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac
