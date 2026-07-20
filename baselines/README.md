# Baselines

This directory is the source-only baseline release for FiSeR. It contains the
two local baseline implementations, source snapshots of the other comparison
methods, and common CUDA training and testing launchers. Trained baseline
weights are not distributed through this repository or a public model release.

```text
baselines/
├── configs/       ResNet-50 and CLIPDetection example configs
├── scripts/       common CUDA training and testing launchers
├── tests/         tests for the local baseline implementations
├── vendor/        third-party source snapshots
├── models.py      ResNet-50 and CLIPDetection definitions
├── train.py       local baseline training entrypoint
├── evaluate.py    local baseline LMDB evaluation entrypoint
└── infer.py       local baseline single-image CUDA inference
```

## Installation

Install the root FiSeR environment first. Third-party methods have incompatible
dependency versions, so install the requirements from the selected directory
under `vendor/` in a separate environment when its upstream implementation
requires them. All common launchers reject a CPU-only PyTorch environment.

Prepared datasets must follow the LMDB schema in [../DATASETS.md](../DATASETS.md).
The same launcher accepts a WildFake, Community Forensics, or other compatible
training LMDB; the repository does not include prepared data.

## Training

Every method is exposed through `scripts/train.sh`. The public example and the
launcher default use five epochs. This is an editable example setting, not a
statement about any private experiment schedule.

```bash
TRAIN_LMDB=/path/to/train.lmdb \
EPOCHS=5 \
OUTPUT_DIR=/path/to/output \
CUDA_VISIBLE_DEVICES=0 \
baselines/scripts/train.sh resnet50
```

Set `NPROC` to the number of visible GPUs for distributed methods and adjust
`BATCH_SIZE`, `NUM_WORKERS`, `VAL_LMDB`, and `RUN_NAME` as needed. Additional
arguments after the method ID are forwarded to that method's native entrypoint.
Generated weights, logs, and metrics stay under `OUTPUT_DIR` (or the ignored
`outputs/baselines/<method-id>` default). Native implementations may add their
own run subdirectory; select the resulting `.pth` or `.pt` file for testing.

SPAI requires its MFM ViT-B/16 initialization checkpoint:

```bash
TRAIN_LMDB=/path/to/train.lmdb \
PRETRAINED=/path/to/mfm_vit_b16.pth \
EPOCHS=5 \
CUDA_VISIBLE_DEVICES=0 \
baselines/scripts/train.sh spai
```

C2P-CLIP accepts either a local CLIP ViT-L/14 directory or its model ID through
`CLIP_MODEL`. AIDE accepts optional local initializers through `RESNET_PATH` and
`CONVNEXT_PATH`.

## Testing

Testing always loads a checkpoint produced by local training. Replace the
method ID with the same ID used for training:

```bash
LMDB_PATH=/path/to/test.lmdb \
MODEL_PATH=/path/to/your_checkpoint.pth \
OUTPUT_DIR=/path/to/evaluation \
CUDA_VISIBLE_DEVICES=0 \
baselines/scripts/evaluate.sh resnet50
```

`MODEL_PATH` is required; the launchers do not download a baseline checkpoint.
Set `EVAL_NAMES` for a display name, and set `NPROC` for methods whose native
evaluation supports distributed execution.

The two local checkpoint formats (`resnet50` and `clip-detection`) also support
single-image CUDA inference:

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.infer \
  --image /path/to/image.jpg \
  --checkpoint /path/to/your_checkpoint.pt \
  --device cuda
```

## Methods

The train and test columns below are complete commands once the environment
variables from the examples above are set.

| Method | ID | Source | Train | Test | Initialization or dependency |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 | `resnet50` | `baselines/` | `baselines/scripts/train.sh resnet50` | `baselines/scripts/evaluate.sh resnet50` | Root FiSeR environment |
| CLIPDetection | `clip-detection` | `baselines/` | `baselines/scripts/train.sh clip-detection` | `baselines/scripts/evaluate.sh clip-detection` | CLIP ViT-L/14 model or local cache |
| CNNDetection | `cnn-detection` | `vendor/CNNDetection-master` | `baselines/scripts/train.sh cnn-detection` | `baselines/scripts/evaluate.sh cnn-detection` | Per-directory requirements |
| LGrad | `lgrad` | `vendor/LGrad-master` | `baselines/scripts/train.sh lgrad` | `baselines/scripts/evaluate.sh lgrad` | Per-directory requirements |
| Gram-Net | `gram-net` | `vendor/Gram-Net-main` | `baselines/scripts/train.sh gram-net` | `baselines/scripts/evaluate.sh gram-net` | Per-directory requirements |
| FreqNet | `freqnet` | `vendor/FreqNet-DeepfakeDetection-main` | `baselines/scripts/train.sh freqnet` | `baselines/scripts/evaluate.sh freqnet` | Per-directory requirements |
| NPR | `npr` | `vendor/NPR-DeepfakeDetection-main` | `baselines/scripts/train.sh npr` | `baselines/scripts/evaluate.sh npr` | Per-directory requirements |
| SAFE | `safe` | `vendor/SAFE-main` | `baselines/scripts/train.sh safe` | `baselines/scripts/evaluate.sh safe` | Per-directory requirements |
| LASTED | `lasted` | `vendor/LASTED` | `baselines/scripts/train.sh lasted` | `baselines/scripts/evaluate.sh lasted` | OpenAI CLIP RN50x64 initialization/cache |
| UniFD | `unifd` | `vendor/UniversalFakeDetect` | `baselines/scripts/train.sh unifd` | `baselines/scripts/evaluate.sh unifd` | OpenAI CLIP initialization/cache |
| AIDE | `aide` | `vendor/AIDE-main` | `baselines/scripts/train.sh aide` | `baselines/scripts/evaluate.sh aide` | Optional `RESNET_PATH` and `CONVNEXT_PATH` |
| DIRE | `dire` | `vendor/DIRE` | `baselines/scripts/train.sh dire` | `baselines/scripts/evaluate.sh dire` | Per-directory requirements |
| Effort | `effort` | `vendor/Effort-AIGI-Detection` | `baselines/scripts/train.sh effort` | `baselines/scripts/evaluate.sh effort` | OpenCLIP ViT-L/14 initialization/cache |
| SPAI | `spai` | `vendor/spai` | `baselines/scripts/train.sh spai` | `baselines/scripts/evaluate.sh spai` | Required `PRETRAINED` MFM ViT-B/16 path |
| C2P-CLIP | `c2p-clip` | `vendor/C2P-CLIP-DeepfakeDetection` | `baselines/scripts/train.sh c2p-clip` | `baselines/scripts/evaluate.sh c2p-clip` | Optional local `CLIP_MODEL` path |
| LOTA | `lota` | `vendor/LOTA` | `baselines/scripts/train.sh lota` | `baselines/scripts/evaluate.sh lota` | Per-directory requirements |
