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

import collections
import pathlib
import random
from functools import partial
from typing import Any, Union, Optional, Iterable
from collections.abc import Callable

import albumentations as A
import torchvision.transforms.functional
from albumentations.augmentations.transforms import ImageCompressionType
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
import cv2
from torchvision.transforms.v2.functional import ten_crop, pad
import filetype

from spai.data import readers
from spai.data import filestorage
from spai import data_utils


class CSVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: pathlib.Path,
        csv_root_path: pathlib.Path,
        split: str,
        transform,
        path_column: str = "image",
        split_column: str = "split",
        class_column: str = "class",
        views: int = 1,
        concatenate_views_horizontally: bool = False,
        lmdb_storage: Optional[pathlib.Path] = None,
        views_generator: Optional[Callable[[Image.Image], tuple[Image.Image, ...]]] = None
    ):
        super().__init__()
        self.csv_path: pathlib.Path = csv_path
        self.csv_root_path: pathlib.Path = csv_root_path
        self.split: str = split
        self.path_column: str = path_column
        self.split_column: str = split_column
        self.class_column: str = class_column
        self.transform = transform
        self.views: int = views
        self.views_generator: Optional[
            Callable[[Image.Image], tuple[Image.Image, ...]]] = views_generator
        self.concatenate_views_horizontally: bool = concatenate_views_horizontally
        self.lmdb_storage: Optional[pathlib.Path] = lmdb_storage

        # Reader to be used for data loading. Its creation is deferred
        self.data_reader: Optional[readers.DataReader] = None

        if split not in ["train", "val", "test"]:
            raise RuntimeError(f"Unsupported split: {split}")

        # Path of the CSV file is expected to be absolute.
        reader = readers.FileSystemReader(pathlib.Path("/"))
        self.entries: list[dict[str, Any]] = reader.read_csv_file(str(self.csv_path))
        self.entries = [e for e in self.entries if e[self.split_column] == self.split]

        self.num_classes: int = len(
            collections.Counter([e[self.class_column] for e in self.entries]).keys()
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray, int]:
        """Returns the requested image sample from the dataset.

        :returns: A tuple containing the image tensor, the labels numpy array and the
         index in dataset.
            Image tensor: (V x 3 x H x W) where V is the number of augmented views.
            Label array:  (1, )
            Index
        """
        # Defer the creation of the data reader until the first read operation in order to
        # properly handle the spawning of multiple processes by DataLoader, where each one
        # should contain a separate reader object.
        if self.data_reader is None:
            self._create_data_reader()

        # Load sample.
        img_obj: Image.Image = self.data_reader.load_image(
            self.entries[idx][self.path_column], channels=3
        )
        label: int = int(self.entries[idx][self.class_column])

        # Generate multiple views of an image either through a provided views generation
        # function or through multiple augmentations of the image.
        if self.views_generator is not None:
            augmented_views: tuple[Image.Image, ...] = self.views_generator(img_obj)
            augmented_views: list[np.ndarray] = [np.array(v) for v in augmented_views]
            augmented_views: list[torch.Tensor] = [
                self.transform(image=v)["image"] for v in augmented_views
            ]
        else:
            img: np.ndarray = np.array(img_obj)
            augmented_views: list[torch.Tensor] = []
            for _ in range(self.views):
                augmented_views.append(self.transform(image=img)["image"])

        # Either concatenate the views in a single big image, or provide them stacked
        # into a new tensor dimension.
        if self.concatenate_views_horizontally:
            augmented_img: torch.Tensor = torch.cat(augmented_views, dim=-1)
            augmented_img = augmented_img.unsqueeze(dim=0)
        else:
            augmented_img: torch.Tensor = torch.stack(augmented_views, dim=0)

        # Cleanup resources.
        img_obj.close()

        return augmented_img, np.array(label, dtype=float), idx

    def get_classes_num(self) -> int:
        return self.num_classes

    def get_dataset_root_path(self) -> pathlib.Path:
        if self.lmdb_storage is not None:
            return self.lmdb_storage
        else:
            return self.csv_root_path

    def update_dataset_csv(
        self,
        column_name: str,
        values: dict[int, Any],
        export_dir: Optional[pathlib.Path] = None
    ) -> None:
        for idx, v in values.items():
            self.entries[idx][column_name] = v

        # Make sure that a valid value for the updated column exists for all entries.
        for e in self.entries:
            if column_name not in e:
                e[column_name] = ""

        if export_dir:
            export_path: pathlib.Path = export_dir / self.csv_path.name
            data_utils.write_csv_file(self.entries, export_path, delimiter=",")

    def _create_data_reader(self) -> None:
        # Limit the number of OpenCV threads to 2 to utilize multiple processes. Otherwise,
        # each process spawns a number of threads equal to the number of logical cores and
        # the overall performance gets worse due to threads congestion.
        cv2.setNumThreads(1)

        if self.lmdb_storage is None:
            self.data_reader: readers.FileSystemReader = readers.FileSystemReader(
                pathlib.Path(self.csv_root_path)
            )
        else:
            self.data_reader: readers.LMDBFileStorageReader = readers.LMDBFileStorageReader(
                filestorage.LMDBFileStorage(self.lmdb_storage, read_only=True)
            )


class CSVDatasetTriplet(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: pathlib.Path,
        csv_root_path: pathlib.Path,
        split: str,
        transform,
        path_column: str = "image",
        split_column: str = "split",
        class_column: str = "class",
        lmdb_storage: Optional[pathlib.Path] = None
    ):
        super().__init__()
        self.csv_path: pathlib.Path = csv_path
        self.csv_root_path: pathlib.Path = csv_root_path
        self.split: str = split
        self.path_column: str = path_column
        self.split_column: str = split_column
        self.class_column: str = class_column
        self.transform = transform
        self.lmdb_storage: Optional[pathlib.Path] = lmdb_storage

        # Reader to be used for data loading. Its creation is deferred
        self.data_reader: Optional[readers.DataReader] = None

        if split not in ["train", "val", "test"]:
            raise RuntimeError(f"Unsupported split: {split}")

        # Path of the CSV file is expected to be absolute.
        reader = readers.FileSystemReader(pathlib.Path("/"))
        self.entries: list[dict[str, Any]] = reader.read_csv_file(str(self.csv_path))
        self.entries = [e for e in self.entries if e[self.split_column] == self.split]

        self.num_classes: int = len(
            collections.Counter([e[self.class_column] for e in self.entries]).keys()
        )

        # Save paths that will be accessed by different dataloaders as numpy arrays in
        # order to avoid copy-on-read of python objects, and thus child processes to
        # take huge amounts of memory.
        self.anchor_v: Optional[np.ndarray] = None
        self.anchor_o: Optional[np.ndarray] = None
        self.positive_v: Optional[np.ndarray] = None
        self.positive_o: Optional[np.ndarray] = None
        self.negative_v: Optional[np.ndarray] = None
        self.negative_o: Optional[np.ndarray] = None
        self.triplets_num: Optional[int] = None
        self.generate_triplets()

    def __len__(self) -> int:
        return self.triplets_num

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the triplet with the specified index.

        :returns: A tuple in the form of (anchor_img, positive_img, negative_img)
        """
        # Defer the creation of the data reader until the first read operation in order to
        # properly handle the spawning of multiple processes by DataLoader, where each one
        # should contain a separate reader object.
        if self.data_reader is None:
            self._create_data_reader()

        anchor_path: str = sequence_to_string(unpack_sequence(self.anchor_v, self.anchor_o, idx))
        positive_path: str = sequence_to_string(
            unpack_sequence(self.positive_v, self.positive_o, idx))
        negative_path: str = sequence_to_string(
            unpack_sequence(self.negative_v, self.negative_o, idx))

        anchor_img_obj: Image.Image = self.data_reader.load_image(anchor_path, channels=3)
        positive_img_obj: Image.Image = self.data_reader.load_image(positive_path, channels=3)
        negative_img_obj: Image.Image = self.data_reader.load_image(negative_path, channels=3)

        anchor_img: np.ndarray = np.array(anchor_img_obj)
        positive_img: np.ndarray = np.array(positive_img_obj)
        negative_img: np.ndarray = np.array(negative_img_obj)

        anchor_img_obj.close()
        positive_img_obj.close()
        negative_img_obj.close()

        return (self.transform(image=anchor_img)["image"],
                self.transform(image=positive_img)["image"],
                self.transform(image=negative_img)["image"])

    def get_classes_num(self) -> int:
        return self.num_classes

    def get_dataset_root_path(self) -> pathlib.Path:
        if self.lmdb_storage is not None:
            return self.lmdb_storage
        else:
            return self.csv_root_path

    def generate_triplets(self) -> None:
        # Separate the entries into groups of each class.
        entries_per_class: dict[int, list[dict[str, Any]]] = {
            i: [] for i in range(self.num_classes)
        }
        for e in self.entries:
            entries_per_class[int(e[self.class_column])].append(e)

        triplets: list[tuple[dict[str,Any], dict[str, Any], dict[str, Any]]] = []
        for class_id, class_group in entries_per_class.items():
            class_group: list[dict[str, Any]] = list(class_group)
            rest_groups: list[list[dict[str, Any]]] = list(entries_per_class.values())
            del rest_groups[class_id]

            for i, e in enumerate(class_group):
                negative_sample: dict[str, Any] = random.choice(random.choice(rest_groups))

                positive_sample: dict[str, Any] = random.choice(class_group)
                while e == positive_sample:
                    positive_sample: dict[str, Any] = random.choice(class_group)

                triplets.append((e, positive_sample, negative_sample))

        self.anchor_v, self.anchor_o = pack_sequences(
            [string_to_sequence(t[0][self.path_column]) for t in triplets]
        )
        self.positive_v, self.positive_o = pack_sequences(
            [string_to_sequence(t[1][self.path_column]) for t in triplets]
        )
        self.negative_v, self.negative_o = pack_sequences(
            [string_to_sequence(t[2][self.path_column]) for t in triplets]
        )
        self.anchor_labels: np.ndarray = np.array([int(t[0][self.class_column]) for t in triplets])
        self.triplets_num = len(triplets)

    def _create_data_reader(self) -> None:
        # Limit the number of OpenCV threads to 2 to utilize multiple processes. Otherwise,
        # each process spawns a number of threads equal to the number of logical cores and
        # the overall performance gets worse due to threads congestion.
        cv2.setNumThreads(1)

        if self.lmdb_storage is None:
            self.data_reader: readers.FileSystemReader = readers.FileSystemReader(
                pathlib.Path(self.csv_root_path)
            )
        else:
            self.data_reader: readers.LMDBFileStorageReader = readers.LMDBFileStorageReader(
                filestorage.LMDBFileStorage(self.lmdb_storage, read_only=True)
            )


def build_loader_finetune(config, logger):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        config.DATA.DATA_PATH,
        config.DATA.CSV_ROOT,
        config=config,
        split_name="train",
        logger=logger
    )
    config.freeze()
    dataset_val, _ = build_dataset(
        config.DATA.DATA_PATH,
        config.DATA.CSV_ROOT,
        config=config,
        split_name="val",
        logger=logger
    )
    logger.info(f"Train images: {len(dataset_train)} | Validation images: {len(dataset_val)}")
    logger.info(f"Train Images Source: {dataset_train.get_dataset_root_path()}")
    logger.info(f"Validation Images Source: {dataset_val.get_dataset_root_path()}")

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        shuffle=True,
        prefetch_factor=config.DATA.PREFETCH_FACTOR
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.VAL_BATCH_SIZE or config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        prefetch_factor=config.DATA.VAL_PREFETCH_FACTOR  or config.DATA.PREFETCH_FACTOR,
        collate_fn=(torch.utils.data.default_collate
                    if not config.MODEL.RESOLUTION_MODE == "arbitrary"
                    else image_enlisting_collate_fn)
    )

    # Setup mixup / cutmix
    mixup_fn = None
    mixup_active: bool = (config.AUG.MIXUP > 0
                          or config.AUG.CUTMIX > 0.
                          or config.AUG.CUTMIX_MINMAX is not None)
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_loader_test(
    config,
    logger,
    split: str = "test",
    dummy_csv_dir: Optional[pathlib.Path] = None,
) -> tuple[list[str], list[torch.utils.data.Dataset], list[torch.utils.data.DataLoader]]:
    # Obtain the root directory for each test input (either a CSV file or a directory).
    input_root_paths: list[pathlib.Path]
    if len(config.DATA.TEST_DATA_CSV_ROOT) > 1:
        input_root_paths = [pathlib.Path(p) for p in config.DATA.TEST_DATA_CSV_ROOT]
    elif len(config.DATA.TEST_DATA_CSV_ROOT) == 1:
        input_root_paths = [pathlib.Path(config.DATA.TEST_DATA_CSV_ROOT[0])
                            for _ in config.DATA.TEST_DATA_PATH]
    else:
        input_root_paths = [pathlib.Path(input_path).parent
                            for input_path in config.DATA.TEST_DATA_PATH]

    # If some input is a directory, create a dummy csv file for it.
    csv_paths: list[pathlib.Path] = []
    csv_root_paths: list[pathlib.Path] = []
    for input_path, input_root_path in zip(config.DATA.TEST_DATA_PATH, input_root_paths):
        input_path: pathlib.Path = pathlib.Path(input_path)
        if input_path.is_dir():
            # Create a dummy csv and point directories
            if dummy_csv_dir is None:
                dummy_csv_dir = pathlib.Path("./outputs")
            entries: list[dict[str, str]] = [
                {
                    "image": str(file_path.name),
                    "split": split,
                    "class": "1"  # TODO: Remove csv requirement for dummy ground-truth.
                }
                for file_path in input_path.iterdir() if filetype.is_image(file_path)
            ]
            dummy_csv_path: pathlib.Path = dummy_csv_dir / f"{input_path.stem}.csv"
            data_utils.write_csv_file(entries, dummy_csv_path, delimiter=",")
            csv_paths.append(dummy_csv_path.absolute())
            csv_root_paths.append(input_path.absolute())  # Paths in CSV are relative to input dir.
        else:
            csv_paths.append(input_path.absolute())
            csv_root_paths.append(input_root_path.absolute())

    # Obtain the separate testing sets and their names.
    test_datasets: list[CSVDataset] = []
    test_datasets_names: list[str] = []
    num_classes_per_dataset: list[int]  = []
    for csv_path, csv_root_path in zip(csv_paths, csv_root_paths):
        csv_path: pathlib.Path = pathlib.Path(csv_path)
        dataset: CSVDataset
        dataset, num_classes = build_dataset(csv_path, csv_root_path, config, split, logger)
        test_datasets.append(dataset)
        test_datasets_names.append(csv_path.stem)
        num_classes_per_dataset.append(num_classes)
    # Check that the number of classes match among all test sets.
    unique_number_of_classes: list[int] = list(collections.Counter(num_classes_per_dataset).keys())
    if len(unique_number_of_classes) > 1:
        raise RuntimeError(
            f"Encountered different number of classes among test sets: {unique_number_of_classes}"
        )

    for dataset, dataset_name in zip(test_datasets, test_datasets_names):
        logger.info(f"Dataset \'{dataset_name}\' | Split: {split} | Total images: {len(dataset)} | "
                    f"Source: {dataset.get_dataset_root_path()}")

    # Create the corresponding data loaders.
    test_data_loaders: list[torch.utils.data.DataLoader] = [
        DataLoader(
            dataset,
            batch_size=config.DATA.TEST_BATCH_SIZE or config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            prefetch_factor=config.DATA.TEST_PREFETCH_FACTOR or config.DATA.PREFETCH_FACTOR,
            collate_fn=(torch.utils.data.default_collate
                        if not config.MODEL.RESOLUTION_MODE == "arbitrary"
                        else image_enlisting_collate_fn)
        )
        for dataset in test_datasets
    ]

    return test_datasets_names, test_datasets, test_data_loaders


def build_dataset(
    csv_path: pathlib.Path,
    csv_root_dir: pathlib.Path,
    config,
    split_name: str,
    logger,
) -> tuple[Union[CSVDataset, CSVDatasetTriplet], int]:
    if split_name not in ["train", "val", "test"]:
        raise RuntimeError(f"Unsupported split: {split_name}")

    transform = build_transform(split_name == "train", config)
    logger.info(f"Data transform | mode: {config.TRAIN.MODE} | split: {split_name}:\n{transform}")

    if split_name == "train" and config.TRAIN.LOSS == "triplet":
        dataset = CSVDatasetTriplet(
            csv_path,
            csv_root_dir,
            split=split_name,
            transform=transform,
            lmdb_storage=pathlib.Path(config.DATA.LMDB_PATH) if config.DATA.LMDB_PATH else None
        )
    elif split_name == "train" and config.TRAIN.LOSS == "supcont":
        assert config.DATA.AUGMENTED_VIEWS > 1, "SupCon loss requires at least 2 views."
        dataset = CSVDataset(
            csv_path,
            csv_root_dir,
            split=split_name,
            transform=transform,
            views=config.DATA.AUGMENTED_VIEWS,
            lmdb_storage=pathlib.Path(config.DATA.LMDB_PATH) if config.DATA.LMDB_PATH else None
        )
    elif split_name == "train" and config.MODEL.RESOLUTION_MODE == "arbitrary":
        dataset = CSVDataset(
            csv_path,
            csv_root_dir,
            split=split_name,
            transform=transform,
            views=config.DATA.AUGMENTED_VIEWS,
            concatenate_views_horizontally=True,
            lmdb_storage=pathlib.Path(config.DATA.LMDB_PATH) if config.DATA.LMDB_PATH else None
        )
    else:
        views_generator: Optional[Callable[[Image.Image], tuple[Image.Image, ...]]]
        if config.TEST.VIEWS_GENERATION_APPROACH == "tencrop":
            def safe_ten_crop(img: Image.Image) -> tuple[Image.Image, ...]:
                width = img.width
                height = img.height
                left_padding: int = max((config.DATA.IMG_SIZE - width) // 2, 0)
                right_padding: int = max(
                    (config.DATA.IMG_SIZE - width) // 2
                    + (((config.DATA.IMG_SIZE - width) % 2) if config.DATA.IMG_SIZE > width else 0),
                    0
                )
                top_padding: int = max((config.DATA.IMG_SIZE - height) // 2, 0)
                bottom_padding: int = max(
                    (config.DATA.IMG_SIZE - height) // 2
                    + (((config.DATA.IMG_SIZE - height) % 2) if config.DATA.IMG_SIZE > height else 0),
                    0
                )
                img = pad(img, [left_padding, top_padding, right_padding, bottom_padding])
                return ten_crop(img, size=config.DATA.IMG_SIZE)

            views_generator = safe_ten_crop
        elif config.TEST.VIEWS_GENERATION_APPROACH is None:
            views_generator = None
        else:
            raise TypeError(f"{config.TEST.VIEW_GENERATION_APPROACH} is not a supported "
                            f"view generation approach.")

        dataset = CSVDataset(
            csv_path,
            csv_root_dir,
            split=split_name,
            transform=transform,
            lmdb_storage=pathlib.Path(config.DATA.LMDB_PATH) if config.DATA.LMDB_PATH else None,
            views_generator=views_generator
        )
    num_classes: int = dataset.get_classes_num()

    return dataset, num_classes


def build_transform(is_train, config) -> Callable[[np.ndarray], np.ndarray]:
    # resize_im: bool = config.DATA.IMG_SIZE > 32
    # # this should always dispatch to transforms_imagenet_train
    # transform = create_transform(
    #     input_size=config.DATA.IMG_SIZE,
    #     is_training=True,
    #     color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
    #     auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
    #     re_prob=config.AUG.REPROB,
    #     re_mode=config.AUG.REMODE,
    #     re_count=config.AUG.RECOUNT,
    #     interpolation=config.DATA.INTERPOLATION,
    # )
    # if not resize_im:
    #     # replace RandomResizedCropAndInterpolation with
    #     # RandomCrop
    #     transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
    # transform.transforms.insert(0, torchvision.transforms.v2.JPEG((50, 100)))
    # transform.transforms.insert(4, torchvision.transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.01, 0.5)))

    if is_train:  # Training augmentations
        transforms_list = []

        if config.AUG.MIN_CROP_AREA == config.AUG.MAX_CROP_AREA:
            transforms_list.append(
                A.PadIfNeeded(min_height=config.DATA.IMG_SIZE, min_width=config.DATA.IMG_SIZE)
            )
            transforms_list.append(
                A.RandomCrop(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE)
            )
        else:
            transforms_list.append(
                A.RandomResizedCrop(size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                    scale=(config.AUG.MIN_CROP_AREA, config.AUG.MAX_CROP_AREA))
            )
        transforms_list.extend([
            A.HorizontalFlip(p=config.AUG.HORIZONTAL_FLIP_PROB),
            A.VerticalFlip(p=config.AUG.VERTICAL_FLIP_PROB),
            A.Rotate(limit=config.AUG.ROTATION_DEGREES,
                     crop_border=True,
                     p=config.AUG.ROTATION_PROB)
        ])
        if config.AUG.ROTATION_PROB > .0:
            # Rotation with crop_border set to True leads to images smaller than the target
            # size. So, restore the target size.
            transforms_list.append(
                A.Resize(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE)
            )
        transforms_list.extend([
            A.GaussianBlur(blur_limit=(3, 9),
                           sigma_limit=(0.01, 0.5),
                           p=config.AUG.GAUSSIAN_BLUR_PROB),
            A.GaussNoise(p=config.AUG.GAUSSIAN_NOISE_PROB),
            A.ColorJitter(
                p=config.AUG.COLOR_JITTER,
                brightness=config.AUG.COLOR_JITTER_BRIGHTNESS_RANGE,
                contrast=config.AUG.COLOR_JITTER_CONTRAST_RANGE,
                saturation=config.AUG.COLOR_JITTER_SATURATION_RANGE,
                hue=config.AUG.COLOR_JITTER_HUE_RANGE,
            ),
            A.Sharpen(p=config.AUG.SHARPEN_PROB,
                      alpha=config.AUG.SHARPEN_ALPHA_RANGE,
                      lightness=config.AUG.SHARPEN_LIGHTNESS_RANGE),
            A.ImageCompression(quality_lower=config.AUG.JPEG_MIN_QUALITY,
                               quality_upper=config.AUG.JPEG_MAX_QUALITY,
                               compression_type=ImageCompressionType.JPEG,
                               p=config.AUG.JPEG_COMPRESSION_PROB),
            A.ImageCompression(quality_lower=config.AUG.WEBP_MIN_QUALITY,
                               quality_upper=config.AUG.WEBP_MAX_QUALITY,
                               compression_type=ImageCompressionType.WEBP,
                               p=config.AUG.WEBP_COMPRESSION_PROB),
        ])
        if config.MODEL.REQUIRED_NORMALIZATION == "imagenet":
            transforms_list.append(
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            )
        elif config.MODEL.REQUIRED_NORMALIZATION == "positive_0_1":
            transforms_list.append(
                A.Normalize(mean=0., std=1.)
            )
        else:
            raise RuntimeError(f"Unsupported Normalization: {config.MODEL.REQUIRED_NORMALIZATION}")
        transforms_list.append(ToTensorV2())
        transform = A.Compose(transforms_list)

    else:  # Inference augmentations
        transforms_list = [
            A.ImageCompression(quality_lower=config.TEST.JPEG_QUALITY,
                               quality_upper=config.TEST.JPEG_QUALITY,
                               compression_type=ImageCompressionType.JPEG,
                               p=1.0 if config.TEST.JPEG_COMPRESSION else .0),
            A.ImageCompression(quality_lower=config.TEST.WEBP_QUALITY,
                               quality_upper=config.TEST.WEBP_QUALITY,
                               compression_type=ImageCompressionType.WEBP,
                               p=1.0 if config.TEST.WEBP_COMPRESSION else .0),
            A.GaussianBlur(blur_limit=(config.TEST.GAUSSIAN_BLUR_KERNEL_SIZE,
                                       config.TEST.GAUSSIAN_BLUR_KERNEL_SIZE),
                           sigma_limit=0,
                           p=1.0 if config.TEST.GAUSSIAN_BLUR else .0),
            A.GaussNoise(var_limit=(config.TEST.GAUSSIAN_NOISE_SIGMA**2,
                                    config.TEST.GAUSSIAN_NOISE_SIGMA**2),
                         p=1.0 if config.TEST.GAUSSIAN_NOISE else .0),
            A.RandomScale(scale_limit=(config.TEST.SCALE_FACTOR-1, config.TEST.SCALE_FACTOR-1),
                          p=1.0 if config.TEST.SCALE else .0)
        ]
        if config.TEST.MAX_SIZE is not None:
            transforms_list.append(A.SmallestMaxSize(max_size=config.TEST.MAX_SIZE))

        if config.TEST.ORIGINAL_RESOLUTION:
            transforms_list.append(A.PadIfNeeded(min_height=config.DATA.IMG_SIZE,
                                                 min_width=config.DATA.IMG_SIZE))
        elif config.TEST.CROP:
            transforms_list.append(A.PadIfNeeded(min_height=config.DATA.IMG_SIZE,
                                                 min_width=config.DATA.IMG_SIZE))
            transforms_list.append(A.CenterCrop(height=config.DATA.IMG_SIZE,
                                                width=config.DATA.IMG_SIZE))
        else:
            transforms_list.append(A.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
        if config.MODEL.REQUIRED_NORMALIZATION == "imagenet":
            transforms_list.append(A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
        elif config.MODEL.REQUIRED_NORMALIZATION == "positive_0_1":
            transforms_list.append(A.Normalize(mean=0., std=1.))
        else:
            raise RuntimeError(f"Unsupported Normalization: {config.MODEL.REQUIRED_NORMALIZATION}")
        transforms_list.append(ToTensorV2())
        transform = A.Compose(transforms_list)

    return transform


def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)


def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])


def pack_sequences(seqs: Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets


def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


def image_enlisting_collate_fn(
    batch: Iterable[tuple[torch.Tensor, np.ndarray, int]]
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Collate function that enlists its entries."""
    return (
        [torch.utils.data.default_collate([s[0]]) for s in batch],
        torch.utils.data.default_collate([s[1] for s in batch]),
        torch.utils.data.default_collate([s[2] for s in batch]),
    )
