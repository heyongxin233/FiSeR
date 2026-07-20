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

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
from tqdm import tqdm
from PIL import Image

from spai import data_utils
from spai.config import get_config
from spai.data import data_finetune, readers


__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "1.0.0"
__revision__: int = 1


@click.command()
@click.option("--cfg", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-c", "--dataset-csv", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-r", "--csv-root-dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output-csv", required=True,
              type=click.Path(dir_okay=False, path_type=Path))
@click.option("-d", "--output-dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--image-column", type=str, default="image", show_default=True)
@click.option("--csv-delimiter", type=str, default=",", show_default=True)
@click.option("--output-csv-delimiter", type=str, default=",", show_default=True)
@click.option("--opt", "extra_options", type=(str, str), multiple=True)
def main(
    cfg: Path,
    dataset_csv: Path,
    csv_root_dir: Optional[Path],
    output_csv: Path,
    output_dir: Path,
    image_column: str,
    csv_delimiter: str,
    output_csv_delimiter: str,
    extra_options: tuple[str, str]
) -> None:
    if csv_root_dir is None:
        csv_root_dir = output_csv.parent

    config = get_config({
        "cfg": str(cfg),
        "opts": extra_options
    })
    config.defrost()
    config.MODEL.REQUIRED_NORMALIZATION = "positive_0_1"
    config.freeze()

    transform: Callable[[np.ndarray], np.ndarray] = data_finetune.build_transform(
        is_train=True, config=config
    )

    entries: list[dict[str, Any]] = data_utils.read_csv_file(dataset_csv, delimiter=csv_delimiter)
    augmented_entries: list[dict[str, Any]] = transform_entries(
        entries, transform, csv_root_dir, output_dir, output_csv, image_column
    )

    data_utils.write_csv_file(augmented_entries, output_csv, delimiter=output_csv_delimiter)
    print(f"Exported CSV to {output_csv}")


def transform_entries(
    entries: list[dict[str, Any]],
    transform: Callable[[np.ndarray], np.ndarray],
    csv_root_dir: Path,
    output_dir: Path,
    output_csv: Path,
    image_column: str
) -> list[dict[str, Any]]:
    augmented_entries: list[dict[str, Any]] = []
    reader: readers.FileSystemReader = readers.FileSystemReader(csv_root_dir)

    for e in tqdm(entries, "Augmenting images", unit="image"):
        img: Image.Image = reader.load_image(e[image_column], channels=3)
        img_data: np.ndarray = np.array(img)
        img.close()
        img_data = transform(image=img_data)["image"]

        out_img_path: Path = output_dir / e[image_column]
        out_img_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray((img_data.detach().permute((1, 2, 0)).cpu().numpy()*255).astype(np.uint8)).save(out_img_path)

        augmented_entry: dict[str, Any] = e.copy()
        augmented_entry[image_column] = str(out_img_path.relative_to(output_csv.parent))
        augmented_entries.append(augmented_entry)

    return augmented_entries


if __name__ == "__main__":
    main()
