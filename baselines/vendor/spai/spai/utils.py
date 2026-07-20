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

import os
import pathlib
import re
from typing import Optional, Iterable

import seaborn as sns
import pandas as pd
import torch
import torch.distributed as dist
import numpy as np
from scipy import interpolate
from matplotlib import colormaps, pyplot as plt
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f">>>>>>>>>> Resuming from {config.MODEL.RESUME} ..........")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def find_pretrained_checkpoints(config) -> list[pathlib.Path]:
    if pathlib.Path(config.PRETRAINED).is_file():
        model_checkpoints: list[pathlib.Path] = [pathlib.Path(config.PRETRAINED)]
    else:
        checkpoints_dir: pathlib.Path = pathlib.Path(config.PRETRAINED)
        model_checkpoints: list[pathlib.Path] = list(checkpoints_dir.glob("ckpt_epoch_*.pth"))
        model_checkpoints.sort(key=lambda p: natural_keys(str(p)))

    return model_checkpoints


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    text = str(text)
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_pretrained(
    config,
    model,
    logger,
    checkpoint_path: Optional[pathlib.Path] = None,
    verbose: bool = True
) -> int:
    if checkpoint_path is None:
        checkpoint_path = pathlib.Path(config.PRETRAINED)

    if verbose:
        logger.info(f">>>>>>>>>> Fine-tuned from {config.PRETRAINED} ..........")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint['model']
    checkpoint_epoch: Optional[int] = checkpoint.get('epoch', None)

    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {
            k.replace('encoder.', ''): v
            for k, v in checkpoint_model.items() if k.startswith('encoder.')
        }
        if verbose:
            logger.info('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        if verbose:
            logger.info('Detect non-pre-trained model, pass without doing anything.')

    if config.MODEL.TYPE == 'swin':
        logger.info(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model, logger)
    elif config.MODEL.TYPE == 'vit':
        logger.info(f">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
        checkpoint = remap_pretrained_keys_vit(model, checkpoint_model, logger)
    else:
        raise NotImplementedError

    msg = model.load_state_dict(checkpoint_model, strict=False)
    if verbose:
        logger.info(msg)
    
    del checkpoint
    torch.cuda.empty_cache()
    if verbose:
        logger.info(f">>>>>>>>>> loaded successfully '{config.PRETRAINED}'")

    return checkpoint_epoch


def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    state_dict = model.state_dict()
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def remap_pretrained_keys_vit(model, checkpoint_model, logger):
    # Duplicate shared rel_pos_bias to each layer
    if getattr(model, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        logger.info("Expand the shared relative position embedding to each transformer block.")
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                logger.info("Position interpolate for %s from %dx%d to %dx%d" % (key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                logger.info("Original positions = %s" % str(x))
                logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias
    
    return checkpoint_model


def remove_imagenet_norm(image: np.ndarray) -> np.ndarray:
    image = image.transpose((1, 2, 0))
    image = image * np.array(IMAGENET_DEFAULT_STD) + np.array(IMAGENET_DEFAULT_MEAN)
    image = image.transpose((2, 0, 1))
    return image


def save_image(
    image: np.ndarray,
    path: pathlib.Path,
    target_size: Optional[tuple[int, int]] = None,
    color_palette: Optional[str] = None,
) -> None:
    """Saves a (C, H, W) ndarray as an image."""
    image = image.transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)

    if color_palette is not None:
        if color_palette != "jet":
            cmap = sns.color_palette(color_palette, as_cmap=True)
        else:
            cmap = colormaps["jet"]
        image = cmap(image)

    pil_image = Image.fromarray(np.squeeze((image*255).astype(np.uint8)))
    if target_size:
        pil_image = pil_image.resize(target_size)
    pil_image.save(path)


def generate_explainability_boxplot(
    scores: np.ndarray,
    without_important_scores: np.ndarray,
    without_non_important_scores: np.ndarray,
    targets: np.ndarray,
    expl_method: str,
    output_path: pathlib.Path,
) -> None:
    plt.clf()
    original_data: pd.DataFrame = pd.DataFrame({
        "method": ["original"]*len(scores),
        "class": targets,
        "score": scores
    })
    without_important_data: pd.DataFrame = pd.DataFrame({
        "method": ["w/o imp."]*len(without_important_scores),
        "class": targets,
        "score": without_important_scores
    })
    without_non_important_data: pd.DataFrame = pd.DataFrame({
        "method": ["w/o n-imp."] * len(without_non_important_scores),
        "class": targets,
        "score": without_non_important_scores
    })
    data = pd.concat([original_data, without_important_data, without_non_important_data])

    box_plot = sns.boxplot(data, x="method", y="score", hue="class")
    box_plot.set_title(expl_method)
    figure = box_plot.get_figure()
    figure.savefig(output_path)


def make_title(s: str) -> str:
    return s.replace("_", " ").title()


def lineplot_2d_multiple_vars(
    x: Iterable[float],
    y: Iterable[Iterable[float]],
    x_name: str,
    y_name: str,
    var_name: Iterable[str],
    title: str,
    output_path: pathlib.Path,
) -> None:
    plt.clf()
    for var_y, var_name in zip(y, var_name):
        plt.plot(x, var_y, label=var_name)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    ax = plt.gca()
    ticks: np.ndarray = np.arange(min(x), max(x), 0.2)
    if max(x) not in ticks:
        ticks = np.append(ticks, max(x))
    plt.xticks(ticks)
    plt.legend()
    figure = plt.gcf()
    figure.savefig(output_path)


def save_image_with_attention_overlay(
    image_patches: torch.Tensor,
    attention_scores: list[float],
    original_height: int,
    original_width: int,
    patch_size: int,
    stride: int,
    overlayed_image_path: pathlib.Path,
    alpha: float = 0.4,
    palette: str = "coolwarm",
    mask_path: Optional[pathlib.Path] = None,
    overlay_path: Optional[pathlib.Path] = None
) -> None:
    """Overlays attention scores over image patches."""
    out_height_blocks: int = ((original_height - patch_size) // stride) + 1
    out_width_blocks: int = ((original_width - patch_size) // stride) + 1
    out_height: int = ((out_height_blocks - 1) * stride) + patch_size
    out_width: int = ((out_width_blocks - 1) * stride) + patch_size

    attention_scores: np.ndarray = np.array(attention_scores)
    # Normalize attention scores in [0, 1].
    # attention_scores = (attention_scores - attention_scores.min()) / attention_scores.max()

    cmap = colormaps[palette]
    # cmap_attention_scores: np.ndarray = cmap(np.array(attention_scores))[:, :3]

    out_image: np.ndarray = np.ones((out_height, out_width, 3))
    overlay: np.ndarray = np.zeros((out_height, out_width))
    overlay_count: np.ndarray = np.zeros_like(overlay)
    for i in range(out_height_blocks):
        for j in range(out_width_blocks):
            out_image[
                i*stride:i*stride+patch_size,
                j*stride:j*stride+patch_size,
                :
            ] = image_patches[0, (i*out_width_blocks)+j].detach().cpu().permute((1, 2, 0)).numpy()
            overlay[
                i*stride:i*stride+patch_size,
                j*stride:j*stride+patch_size
            ] += attention_scores[(i*out_width_blocks)+j]
            overlay_count[
                i * stride:i * stride + patch_size,
                j * stride:j * stride + patch_size
            ] += 1
    overlay = overlay / overlay_count
    overlay = (overlay - overlay.min()) / overlay.max()

    colormapped_overlay = cmap(overlay)[:, :, :3]

    overlayed_image: np.ndarray = (1 - alpha) * out_image + alpha * colormapped_overlay
    overlayed_image = (overlayed_image * 255).astype(np.uint8)

    Image.fromarray(overlayed_image).save(overlayed_image_path)
    if mask_path is not None:
        Image.fromarray((overlay*255).astype(np.uint8)).save(mask_path)
    if overlay_path is not None:
        Image.fromarray((colormapped_overlay*255).astype(np.uint8)).save(overlay_path)


def inf_nan_to_num(value, nan_value, inf_value):
    if np.isinf(value):
        return inf_value
    elif np.isnan(value):
        return nan_value
    else:
        return value
