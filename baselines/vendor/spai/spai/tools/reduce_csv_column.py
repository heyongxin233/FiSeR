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

"""Tool for conditionally reducing a column of a CSV file."""
from collections.abc import Callable
from pathlib import Path
from statistics import stdev, mean
from typing import Optional, Any

import click

from spai import data_utils


__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "1.0.0"
__revision__: int = 1


MERGE_OPERATIONS: list[str] = [
    "avg",
    "std",
    "count"
]


@click.command(help="Conditionally reduces a csv column.")
@click.option("-c", "--csv-path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to the csv file.")
@click.option("-d", "--csv-delimiter", type=str, default=",")
@click.option("-p", "--reduced-csv", required=True,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Path to the new csv file. It will include all the columns of the input csv "
                   "and the new columns specified by `--frame-num-column` and `frame-column`.")
@click.option("-r", "--reduce-operation", type=str, default="avg")
@click.option("-m", "--reduce-column", required=True, type=str)
@click.option("-l", "--reduce-into-column", type=str)
@click.option("-k", "--condition-column", required=True, type=str)
def merge_scores(
    csv_path: Path,
    csv_delimiter: str,
    reduced_csv: Path,
    reduce_operation: str,
    reduce_column: str,
    reduce_into_column: Optional[str],
    condition_column: str
) -> None:
    entries: list[dict[str, Any]] = data_utils.read_csv_file(csv_path, delimiter=csv_delimiter)

    reduction_groups: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        reduction_groups[e[condition_column]] = reduction_groups.get(e[condition_column], []) + [e]

    reduced_entries: list[dict[str, Any]] = []
    reduce_function: Callable[[list[float]], float] = get_reduce_function(reduce_operation)
    if reduce_into_column is None:
        reduce_into_column = reduce_column
    for g in reduction_groups.values():
        reduction_values: list[float] = [float(e[reduce_column]) for e in g]
        reduced_value: float = reduce_function(reduction_values)
        reduced_entry: dict[str, Any] = g[0].copy()
        del reduced_entry[reduce_column]
        reduced_entry[reduce_into_column] = reduced_value
        reduced_entries.append(reduced_entry)

    data_utils.write_csv_file(reduced_entries, reduced_csv, delimiter=csv_delimiter)


def get_reduce_function(name: str) -> Callable[[list[float]], float]:
    if name == "avg":
        return mean
    elif name == "std":
        return lambda l: stdev(l) if len(l) > 1 else 0
    elif name == "count":
        return len
    else:
        raise ValueError(f"Unknown reduce_operation: {name}")


if __name__ == '__main__':
    merge_scores()
