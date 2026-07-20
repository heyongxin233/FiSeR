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

from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .sid import build_cls_vit, build_mf_vit
from .mfm import build_mfm


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model


def build_cls_model(config):
    model_type = config.MODEL.TYPE
    task_type = config.MODEL.SID_APPROACH
    if model_type == "vit" and task_type == "single_extraction":
        model = build_cls_vit(config)
    elif model_type == "vit" and task_type == "freq_restoration":
        model = build_mf_vit(config)
    else:
        raise NotImplementedError(f"Unknown cls model: {model_type}")
    return model
