#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD="${1:?Usage: $0 METHOD}"
shift

: "${LMDB_PATH:?Set LMDB_PATH to a prepared evaluation LMDB}"
: "${MODEL_PATH:?Set MODEL_PATH to a checkpoint produced by local training}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "CUDA is required")'

NPROC="${NPROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/baselines/${METHOD}}"
EVAL_NAMES="${EVAL_NAMES:-evaluation}"
export BASELINE_LOG_DIR="${BASELINE_LOG_DIR:-${OUTPUT_DIR}/logs}"
mkdir -p "${OUTPUT_DIR}"

case "${METHOD}" in
  resnet50|clip-detection)
    cd "${ROOT}"
    exec torchrun --standalone --nproc_per_node="${NPROC}" -m baselines.evaluate \
      --checkpoint "${MODEL_PATH}" --dataset-lmdb "${LMDB_PATH}" \
      --output "${OUTPUT_DIR}/metrics.json" --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" "$@"
    ;;
  cnn-detection)
    cd "${ROOT}/baselines/vendor/CNNDetection-master"
    exec python eval.py --model_path "${MODEL_PATH}" --lmdb_path "${LMDB_PATH}" \
      --eval_names "${EVAL_NAMES}" --batch_size "${BATCH_SIZE}" \
      --num_threads "${NUM_WORKERS}" "$@"
    ;;
  lgrad)
    cd "${ROOT}/baselines/vendor/LGrad-master/CNNDetection"
    exec torchrun --standalone --nproc_per_node="${NPROC}" test.py \
      --distributed --model_path "${MODEL_PATH}" --lmdb_path "${LMDB_PATH}" \
      --eval_names "${EVAL_NAMES}" --batch_size "${BATCH_SIZE}" \
      --num_threads "${NUM_WORKERS}" "$@"
    ;;
  gram-net)
    cd "${ROOT}/baselines/vendor/Gram-Net-main"
    exec python test.py --model_path "${MODEL_PATH}" --lmdb_path "${LMDB_PATH}" \
      --eval_names "${EVAL_NAMES}" --batch_size "${BATCH_SIZE}" \
      --num_threads "${NUM_WORKERS}" "$@"
    ;;
  freqnet)
    cd "${ROOT}/baselines/vendor/FreqNet-DeepfakeDetection-main"
    exec torchrun --standalone --nproc_per_node="${NPROC}" test.py \
      --model_path "${MODEL_PATH}" --lmdb_path "${LMDB_PATH}" \
      --batch_size "${BATCH_SIZE}" --num_threads "${NUM_WORKERS}" "$@"
    ;;
  npr)
    cd "${ROOT}/baselines/vendor/NPR-DeepfakeDetection-main"
    exec python test.py --model_path "${MODEL_PATH}" --lmdb_path "${LMDB_PATH}" \
      --eval_names "${EVAL_NAMES}" --batch_size "${BATCH_SIZE}" \
      --num_threads "${NUM_WORKERS}" "$@"
    ;;
  safe)
    cd "${ROOT}/baselines/vendor/SAFE-main"
    exec python main_finetune.py --eval true --model SAFE \
      --resume "${MODEL_PATH}" --data_path "${LMDB_PATH}" \
      --eval_data_path "${LMDB_PATH}" --output_dir "${OUTPUT_DIR}" \
      --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" "$@"
    ;;
  lasted)
    cd "${ROOT}/baselines/vendor/LASTED"
    exec python main.py --isTrain 0 --test_lmdb "${LMDB_PATH}" \
      --eval_names "${EVAL_NAMES}" --resume "${MODEL_PATH}" \
      --batch_size "${BATCH_SIZE}" "$@"
    ;;
  unifd)
    cd "${ROOT}/baselines/vendor/UniversalFakeDetect"
    exec torchrun --standalone --nproc_per_node="${NPROC}" test.py \
      --distributed --model_path "${MODEL_PATH}" --lmdb_path "${LMDB_PATH}" \
      --eval_names "${EVAL_NAMES}" --batch_size "${BATCH_SIZE}" \
      --num_threads "${NUM_WORKERS}" "$@"
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
    exec python main_finetune.py --eval true --model AIDE \
      --resume "${MODEL_PATH}" --data_path "${LMDB_PATH}" \
      --eval_data_path "${LMDB_PATH}" --output_dir "${OUTPUT_DIR}" \
      --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
      "${AIDE_INIT_ARGS[@]}" "$@"
    ;;
  dire)
    cd "${ROOT}/baselines/vendor/DIRE"
    exec python test.py --gpus 0 --ckpt "${MODEL_PATH}" \
      exp_root "${OUTPUT_DIR}" lmdb_path "${LMDB_PATH}" \
      eval_names "${EVAL_NAMES}" batch_size "${BATCH_SIZE}" \
      num_workers "${NUM_WORKERS}" metrics_out "${OUTPUT_DIR}/metrics.json" "$@"
    ;;
  effort)
    cd "${ROOT}/baselines/vendor/Effort-AIGI-Detection/UniversalFakeDetect_Benchmark"
    exec torchrun --standalone --nproc_per_node="${NPROC}" test.py \
      --distributed --compile --use_svd --arch CLIP:ViT-L/14_svd \
      --lmdb_path "${LMDB_PATH}" --model_path "${MODEL_PATH}" \
      --batch_size "${BATCH_SIZE}" --num_threads "${NUM_WORKERS}" "$@"
    ;;
  spai)
    cd "${ROOT}/baselines/vendor/spai"
    exec torchrun --standalone --nproc_per_node="${NPROC}" test.py \
      --lmdb_path "${LMDB_PATH}" --model_path "${MODEL_PATH}" \
      --output "${OUTPUT_DIR}" --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" --precision "${PRECISION:-bf16-mixed}" "$@"
    ;;
  c2p-clip)
    cd "${ROOT}/baselines/vendor/C2P-CLIP-DeepfakeDetection"
    exec torchrun --standalone --nproc_per_node="${NPROC}" test.py \
      --distributed --amp --amp_dtype "${AMP_DTYPE:-bf16}" \
      --lmdb_path "${LMDB_PATH}" --model_path "${MODEL_PATH}" \
      --eval_names "${EVAL_NAMES}" --batch_size "${BATCH_SIZE}" \
      --num_threads "${NUM_WORKERS}" \
      --clip "${CLIP_MODEL:-openai/clip-vit-large-patch14}" \
      --lora_r 6 --lora_alpha 6 --lora_dropout 0.5 "$@"
    ;;
  lota)
    cd "${ROOT}/baselines/vendor/LOTA"
    exec torchrun --standalone --nproc_per_node="${NPROC}" test.py \
      --distributed --lmdb_path "${LMDB_PATH}" --model_path "${MODEL_PATH}" \
      --eval_names "${EVAL_NAMES}" --results_dir "${OUTPUT_DIR}" \
      --val_batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
      --precision "${PRECISION:-bf16}" --bit_mode scaling --patch_size 32 \
      --patch_mode max "$@"
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac
