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

"""Script for generating a CSV file for the Synthbuster dataset."""
from pathlib import Path
import random
from typing import Any, Optional

import click

import data_utils


__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "1.0.0"
__revision__: int = 1


@click.command()
@click.option("--synthbuster_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--raise_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output_csv",
              type=click.Path(dir_okay=False, path_type=Path),
              required=True)
@click.option("-r", "--csv_root_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-d", "--output_csv_delimiter", type=str, default=",", show_default=True)
@click.option("-n", "--samples_num", type=int, default=None, show_default=True)
@click.option("-f", "--filter", type=str, multiple=True)
def main(
    synthbuster_dir: Optional[Path],
    raise_dir: Optional[Path],
    output_csv: Path,
    csv_root_dir: Optional[Path],
    output_csv_delimiter: str,
    samples_num: Optional[int],
    filter: list[str]
) -> None:
    if csv_root_dir is None:
        csv_root_dir = output_csv.parent

    entries: list[dict[str, Any]] = []
    # valid_num: int = 0

    synthbuster_data_gen = synthbuster_dir.rglob("*")
    for p in synthbuster_data_gen:
        # if filetype.is_image(p):
        if p.is_file() and p.suffix == ".png":
            path_parts: list[str] = p.parts

            filter_found: bool = False if len(filter) > 0 else True
            for f in filter:
                if f in path_parts:
                    filter_found = True
                    break
            if not filter_found:
                continue

            entries.append({
                "image": str(p.relative_to(csv_root_dir)),
                "class": 1,
                "split": "test"
            })
            # valid_num += 1

    prompts_csv: Path = synthbuster_dir / "prompts.csv"
    prompts_entries: list[dict[str, Any]] = data_utils.read_csv_file(prompts_csv)
    raise_name_column: str = "image name (matching Raise-1k)"
    for e in prompts_entries:
        raise_name = e[raise_name_column]
        raise_img: Path = raise_dir / "tiff" / f"{raise_name}.tif"
        assert raise_img.exists()
        entries.append({
            "image": str(raise_img.relative_to(csv_root_dir)),
            "class": 0,
            "split": "test"
        })

    if samples_num is not None:
        entries = random.sample(entries, samples_num)

    data_utils.write_csv_file(entries, output_csv, delimiter=output_csv_delimiter)


if __name__ == "__main__":
    main()
