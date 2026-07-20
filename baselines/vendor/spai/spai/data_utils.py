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
import csv
import hashlib
from typing import Any, Optional


def read_csv_file(
    path: pathlib.Path,
    delimiter: str = ","
) -> list[dict[str, Any]]:
    with path.open("r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        contents: list[dict[str, Any]] = [row for row in reader]
    return contents


def write_csv_file(
    data: list[dict[str, Any]],
    output_file: pathlib.Path,
    fieldnames: Optional[list[str]] = None,
    delimiter: str = "|"
) -> None:
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    with output_file.open("w", newline="") as f:
        writer: csv.DictWriter = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for r in data:
            writer.writerow(r)


def compute_file_md5(path: pathlib.Path) -> str:
    with path.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()
