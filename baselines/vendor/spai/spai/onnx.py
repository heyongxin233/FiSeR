# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib

import numpy as np
import torch
import torch.nn
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from spai.models import filters


def compare_pytorch_onnx_models(
    pytorch_model: torch.nn.Module,
    onnx_encoder: pathlib.Path,
    onnx_patch_aggregator: pathlib.Path,
    includes_preprocessing: bool = True,
    device: str = "cpu"
) -> None:
    import onnxruntime

    test_input: torch.Tensor = torch.rand(1, 3, 512, 512, device=device)

    # Compute model's output using ONNX RT.
    ort_encoder_session = onnxruntime.InferenceSession(
        onnx_encoder,
        providers=["CUDAExecutionProvider" if "cuda" in device else "CPUExecutionProvider"]
    )
    ort_aggregator_session = onnxruntime.InferenceSession(
        onnx_patch_aggregator,
        providers=["CPUExecutionProvider" if "cuda" in device else "CPUExecutionProvider"]
    )
    onnx_pred: torch.Tensor = predict_onnx_model(
        ort_encoder_session, ort_aggregator_session, test_input,
        includes_preprocessing=includes_preprocessing, encoder_batch_size=4,
        device=torch.device(device)
    )

    # Compute model's output using the pytorch model.
    pytorch_pred: torch.Tensor = pytorch_model(test_input)
    pytorch_pred = torch.sigmoid(pytorch_pred)

    print(f"ONNX Overall Prediction: {to_numpy(onnx_pred).item()}")
    print(f"PyTorch Overall Prediction: {to_numpy(pytorch_pred).item()}")

    # Test only the patch aggregator with a new random input.
    aggr_input: torch.Tensor = torch.randn(1, 4, 1096)

    onnx_pred: torch.Tensor = predict_onnx_aggregator(ort_aggregator_session, aggr_input)

    # Compute aggregators's output using the pytorch model.
    aggr_input = aggr_input.to(device)
    pytorch_pred: torch.Tensor = pytorch_model.patches_attention(aggr_input)
    pytorch_pred = pytorch_model.norm(pytorch_pred)
    pytorch_pred = pytorch_model.cls_head(pytorch_pred)
    pytorch_pred = torch.sigmoid(pytorch_pred)

    print(f"ONNX Aggregator Prediction: {to_numpy(onnx_pred).item()}")
    print(f"PyTorch Overall Prediction: {to_numpy(pytorch_pred).item()}")


def predict_onnx_model(
    encoder_session,
    aggregator_session,
    test_input: torch.Tensor,
    includes_preprocessing: bool = True,
    encoder_batch_size: int = 1,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    from spai.models import utils as model_utils
    patched_input: torch.Tensor = model_utils.patchify_image(
        test_input,
        (224, 224),
        (224, 224)
    )  # B x L x C x H x W
    patched_input = patched_input.squeeze(dim=0)

    if includes_preprocessing:
        patch_features: list[torch.Tensor] = []
        for i in range(patched_input.size(0)):
            encoder_input = {k.name: to_numpy(v) for k, v in
                             zip(encoder_session.get_inputs(), (patched_input[i:i + 1, :, :, :],))}
            encoder_output = encoder_session.run(None, encoder_input)
            patch_features.extend(encoder_output)
        patch_features: np.ndarray = np.expand_dims(np.concatenate(patch_features, axis=0), axis=0)
    else:
        patch_features: list[torch.Tensor] = []
        for i in range(0, patched_input.size(0), encoder_batch_size):
            batched_input: torch.Tensor = patched_input[i:i+encoder_batch_size, :, :, :]
            x, x_low, x_high = fft_preprocessing(batched_input, device=device)
            encoder_input: dict[str, np.ndarray] = {
                "x": to_numpy(x),
                "x_low": to_numpy(x_low),
                "x_high": to_numpy(x_high)
            }
            encoder_output = encoder_session.run(None, encoder_input)
            patch_features.extend(encoder_output)
        patch_features: np.ndarray = np.expand_dims(np.concatenate(patch_features, axis=0), axis=0)

    aggregator_input = {k.name: v for k, v in
                        zip(aggregator_session.get_inputs(), (patch_features,))}

    aggregator_output = aggregator_session.run(None, aggregator_input)
    onnx_pred: torch.Tensor = torch.sigmoid(torch.from_numpy(aggregator_output[0]))
    return onnx_pred


def predict_onnx_aggregator(
    aggregator_session,
    aggr_input: torch.Tensor
) -> torch.Tensor:
    aggregator_input = {k.name: to_numpy(v) for k, v in
                        zip(aggregator_session.get_inputs(), (aggr_input,))}
    aggregator_output = aggregator_session.run(None, aggregator_input)
    onnx_pred: torch.Tensor = torch.sigmoid(torch.from_numpy(aggregator_output[0]))
    return onnx_pred


def fft_preprocessing(
    x: torch.Tensor,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frequencies_mask: torch.Tensor = filters.generate_circular_mask(
        224, 16, device=device
    )
    x_low, x_hi = filters.filter_image_frequencies(x.float(), frequencies_mask)
    x_low = torch.clamp(x_low, min=0., max=1.).to(x.dtype)
    x_hi = torch.clamp(x_hi, min=0., max=1.).to(x.dtype)
    # Normalize all components according to ImageNet.
    backbone_norm = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    x = backbone_norm(x)
    x_low = backbone_norm(x_low)
    x_hi = backbone_norm(x_hi)
    return x, x_low, x_hi


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
