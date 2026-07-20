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

from pathlib import Path
from typing import Union

import torch
import tsnecuda
import matplotlib.pyplot as plt


class Hook:
    def __init__(self, name, module, aggregator: dict[str, list[torch.Tensor]]):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)
        self.aggregator = aggregator

    def hook_fn(self, module, input, output):
        aggregator = self.aggregator.get(self.name, [])
        aggregator.append(output.detach().cpu())
        self.aggregator[self.name] = aggregator

    def close(self):
        self.hook.remove()


@torch.no_grad()
def visualize_tsne(
    config,
    data_loader,
    data_name,
    model,
    neptune_run,
    verbose: bool = True,
):
    model.eval()
    embeddings: dict[str, Union[list[torch.Tensor], torch.Tensor]] = {}
    sca_embeddings: dict[str, Union[list[torch.Tensor], torch.Tensor]] = {}
    targets: list[torch.Tensor] = []
    sca_targets: list[torch.Tensor] = []

    for idx, (images, target, dataset_idx) in enumerate(data_loader):
        assert isinstance(images, list)
        assert len(images) == 1, "Batch size required to be set to 1."

        hook = Hook(str(dataset_idx.item()), model.mfvit.features_processor, embeddings)
        sca_hook = Hook(str(dataset_idx.item()), model.to_out, sca_embeddings)

        # In case of arbitrary resolution models the batch is provided as a list of tensors.
        images = [img.cuda(non_blocking=True) for img in images]
        # Remove views dimension. Always 1 during inference.
        images = [img.squeeze(dim=1) for img in images]
        # Compute output.
        model(images, config.MODEL.FEATURE_EXTRACTION_BATCH)

        hook.close()
        sca_hook.close()
        embeddings[str(dataset_idx.item())] = torch.cat(embeddings[str(dataset_idx.item())], dim=0)
        sca_embeddings[str(dataset_idx.item())] = torch.cat(
            sca_embeddings[str(dataset_idx.item())], dim=0
        )
        targets.append(target.repeat(embeddings[str(dataset_idx.item())].size(0)))
        sca_targets.append(target)

    embeddings: torch.Tensor = torch.cat(
        [embeddings[str(i)] for i in range(len(embeddings))], dim=0
    )
    targets: torch.Tensor = torch.cat(targets, dim=0).long()
    sca_embeddings: torch.Tensor = torch.cat(
        [sca_embeddings[str(i)] for i in range(len(sca_embeddings))], dim=0
    ).squeeze(dim=1)
    sca_targets: torch.Tensor = torch.cat(sca_targets, dim=0).long()

    for perplexity in [10 * i for i in range(2, 11)]:
        # Embed the spectral vectors for each patch of the image.
        tsne_embed = tsnecuda.TSNE(perplexity=perplexity).fit_transform(embeddings.numpy())
        plt.figure(figsize=(5, 5))
        plt.scatter(tsne_embed[:, 1], tsne_embed[:, 0], c=targets.numpy(),
                    cmap='PiYG', marker='.')
        plot_file: Path = (Path(config.OUTPUT) / "embeds_viz"
                           / f"{data_name}_spectral_vectors_p={perplexity}.png")
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file)
        plt.close()

        # Embed the values of spectral reconstruction similarity.
        tsne_embed = tsnecuda.TSNE(perplexity=perplexity).fit_transform(embeddings[:, :72].numpy())
        plt.figure(figsize=(5, 5))
        plt.scatter(tsne_embed[:, 1], tsne_embed[:, 0], c=targets.numpy(),
                    cmap='PiYG', marker='.')
        plot_file: Path = (Path(config.OUTPUT) / "embeds_viz"
                           / f"{data_name}_srs_p={perplexity}.png")
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file)
        plt.close()

        # Embed the spectral vectors for each patch of the image.
        tsne_embed = tsnecuda.TSNE(perplexity=perplexity).fit_transform(embeddings[:, 72:].numpy())
        plt.figure(figsize=(5, 5))
        plt.scatter(tsne_embed[:, 1], tsne_embed[:, 0], c=targets.numpy(),
                    cmap='PiYG', marker='.')
        plot_file: Path = (Path(config.OUTPUT) / "embeds_viz"
                           / f"{data_name}_scv_p={perplexity}.png")
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file)
        plt.close()

        # Embed image-level spectral vectors for each patch of the image.
        tsne_embed = tsnecuda.TSNE(perplexity=perplexity).fit_transform(sca_embeddings.numpy())
        plt.figure(figsize=(5, 5))
        plt.scatter(tsne_embed[:, 1], tsne_embed[:, 0], c=sca_targets.numpy(),
                    cmap='PiYG', marker='.')
        plot_file: Path = (Path(config.OUTPUT) / "embeds_viz"
                           / f"{data_name}_image_features_p={perplexity}.png")
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file)
        plt.close()
