---
library_name: transformers
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
base_model: facebook/dinov3-vitl16-pretrain-lvd1689m
tags:
- image-classification
- ai-generated-image-detection
- dinov3
pipeline_tag: image-feature-extraction
---

# FiSeR DINOv3 ViT-L/16

This model is the image encoder trained by FiSeR with hierarchical supervised
contrastive learning. It outputs a 1024-dimensional CLS representation. The
released model is a backbone; attach a kNN, prototype, linear, or SVM head for
the binary natural/synthetic decision described in the paper.

The released weights correspond to the `wild_model19.pth` checkpoint from
zero-indexed epoch 19 (the 20th and final training epoch). The training-only
source-center parameter is omitted from this backbone export.

Paper: [arXiv:2606.00606](https://arxiv.org/abs/2606.00606)

Code: [heyongxin233/FiSeR](https://github.com/heyongxin233/FiSeR)

## Usage

```python
from transformers import AutoImageProcessor, AutoModel
import torch

model_id = "heyongxin233/FiSeR-DINOv3-ViT-L16"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
image = ...  # PIL.Image.Image
inputs = processor(images=image, return_tensors="pt")
with torch.inference_mode():
    features = model(**inputs).last_hidden_state[:, 0]
```

Use the FiSeR repository's `evaluate.py` to reproduce the paper's GPU-FAISS
retrieval protocol. The published safetensors SHA-256 is
`3bfda6c3040fe22f8fe82588bd4c8506572a6844a04c15bb5c8feebbbd9e11a5`.

For a binary decision on one image, load this backbone together with an
explicit kNN feature bank or a portable linear/prototype head:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --image image.jpg \
  --model-name heyongxin233/FiSeR-DINOv3-ViT-L16 \
  --head knn --database-pt wildfake_train_embeddings.pt \
  --layer 18 --k 25 --faiss-devices 0
```

`scripts/build_inference_head.py` creates a small linear/prototype head from a
training embedding archive, and historical layered `linear_probes.pt` files
are accepted directly with `--head linear --layer 18`. Head probabilities are
not a calibrated measure of
provenance confidence; use the output as a detector score under the documented
dataset protocol.

The fixed-layer direct-transfer average is 98.23 AUROC / 93.43 TPR@5% FPR over
WildFake, Community Forensics, AIGIBench, Chameleon, and GenImage. Refer to the
repository for dataset revisions and the exact per-dataset protocol. This
checkpoint must not be used as a substitute for content provenance, safety
review, or human moderation.

## Training data and limitations

The model was trained on the WildFake training split. It can be sensitive to
dataset composition, image post-processing, and generator families absent from
WildFake. Dataset and DINOv3 licensing terms remain applicable.
