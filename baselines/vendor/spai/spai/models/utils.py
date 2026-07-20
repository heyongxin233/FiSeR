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

# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def patchify_image(
    img: torch.Tensor,
    patch_size: tuple[int, int],
    stride: tuple[int, int]
) -> torch.Tensor:
    """Splits an input image into patches,

    :param img: Input image of size (B, C, H, W).
    :param patch_size: (height, width) of patches.
    :param stride: Stride on (height, width) dimensions.

    :returns: Patchified image of size (B, L, C, patch_height, patch_width).
    """
    kh, kw = patch_size
    dh, dw = stride
    img = img.unfold(2, kh, dh).unfold(3, kw, dw)
    img = img.permute(0, 2, 3, 1, 4, 5)
    img = img.contiguous()
    img = img.view(img.size(0), -1, img.size(3), kh, kw)
    return img


class ExportableImageNormalization(nn.Module):
    """Layer that normalizes an image, like `torchvision.transforms.Normalize`.

    The intention of building this class is that `torchvision.transforms.Normalize`
    includes conditional logic that exporters cannot export,
    like torch.onnx.dynamo_export.
    Thus, this class practically applies
    an (`x` - `mean`) / `std `operation without further checks.
    """

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.mean: tuple[float, ...] = mean
        self.std: tuple[float, ...] = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: A tensor of shape B x C x H x W.
        """
        dtype = x.dtype
        device = x.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std, dtype=dtype, device=device)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        return x.sub_(mean).div_(std)


def exportable_std(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard deviation operation exportable to ONNx.

    As of PyTorch 2.5 the dynamo exporter lacks support for the
    `torch.ops.prims.var.default` operation. Therefore, it fails
    to export variance operations applied to tensor.
    Github issue: https://github.com/pytorch/pytorch/issues/141969
    """
    m: torch.Tensor = x.mean(dim, keepdim=True)
    return  torch.sqrt(torch.pow(x-m, 2).sum(dim) / (x.size(dim) - 1)) # apply Bessel’s correction
