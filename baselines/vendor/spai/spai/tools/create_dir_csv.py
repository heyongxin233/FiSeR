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

"""Script for generating a CSV file from an images' directory."""
from pathlib import Path
import random
from typing import Any, Optional

import click

from spai import data_utils


__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "2.0.0"
__revision__: int = 2


@click.command()
@click.option("--train_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--val_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--test_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output_csv",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True)
@click.option("-r", "--csv_root_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-d", "--output_csv_delimiter", type=str, default=",", show_default=True)
@click.option("-n", "--samples_num", type=int, default=None, show_default=True)
@click.option("--recursive", is_flag=True, default=False, show_default=True)
@click.option("-f", "--filter", type=str, multiple=True)
def main(
    train_dir: Optional[Path],
    val_dir: Optional[Path],
    test_dir: Optional[Path],
    output_csv: Path,
    csv_root_dir: Optional[Path],
    output_csv_delimiter: str,
    samples_num: Optional[int],
    recursive: bool,
    filter: list[str]
) -> None:
    if csv_root_dir is None:
        csv_root_dir = output_csv.parent

    entries: list[dict[str, Any]] = []
    # valid_num: int = 0

    split_dirs: list[Path] = []
    split_labels: list[str] = []
    if train_dir is not None:
        split_dirs.append(train_dir)
        split_labels.append("train")
    if val_dir is not None:
        split_dirs.append(val_dir)
        split_labels.append("val")
    if test_dir is not None:
        split_dirs.append(test_dir)
        split_labels.append("test")

    for s_dir, s_label in zip(split_dirs, split_labels):
        data_gen = s_dir.rglob("*")
        for p in data_gen:
            # if filetype.is_image(p):
            if p.is_file():
                path_parts: list[str] = p.parts

                filter_found: bool = False if len(filter) > 0 else True
                for f in filter:
                    if f in path_parts:
                        filter_found = True
                        break
                if not filter_found:
                    continue

                # Find label
                if "0_real" in path_parts:
                    label: int = 0
                elif "1_fake" in path_parts:
                    label: int = 1
                else:
                    raise RuntimeError(f"No valid class identifier found in image path: {p}")

                entries.append({
                    "image": str(p.relative_to(csv_root_dir)),
                    "class": label,
                    "split": s_label
                })
                # valid_num += 1

    if samples_num is not None:
        entries = random.sample(entries, samples_num)

    data_utils.write_csv_file(entries, output_csv, delimiter=output_csv_delimiter)


if __name__ == "__main__":
    main()
