# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Dataset implementation for the TSN module"""
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from mindspore import dataset as de
from mindspore.dataset.vision.py_transforms import HWC2CHW

from src.transforms import GroupCompose
from src.transforms import GroupMultiScaleCrop
from src.transforms import GroupNormalize
from src.transforms import GroupOverSample
from src.transforms import GroupRandomHorizontalFlip
from src.transforms import Stack
from src.utils import get_frames_combinations

_PREPROCESS_MEAN = (104, 117, 128)
_PREPROCESS_STD = (1, 1, 1)


class VideoRecord:
    """Information about a single video folder"""

    def __init__(self, row):
        if not isinstance(row, str):
            raise TypeError(f'Invalid type of row: {type(row)}. Must be string.')

        data = list(row.strip().split(' '))
        if len(data) != 3:
            raise ValueError(f'Invalid data format: {data}. Required format: "<video name> <num frames> <label>"')

        self._name = data[0]
        self._num_frames = int(data[1])
        self._label = int(data[2])

        if self._num_frames < 3:  # for not jester dataset
            raise ValueError(f'Expected number of frames to be non-less than 3, got {self._num_frames}.')

    @property
    def name(self):
        """Video folder name"""
        return self._name

    @property
    def num_frames(self):
        """Number of frames in video"""
        return self._num_frames

    @property
    def label(self):
        """Label of video"""
        return self._label


class TSNDataSet:
    """TSNDataset"""

    def __init__(self, root_path, list_file, num_segments=3,
                 image_tmpl='img_{:05d}.jpg', transforms=None,
                 random_shift=True, test_mode=False, check_frames=False):

        self.root_path = Path(root_path).resolve()
        self.num_segments = num_segments
        self.transforms = transforms
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode
        self._check_frames = check_frames

        self.video_list = self._parse_list(self.root_path, list_file)

    def _load_image(self, name_video, frame_idx):
        """Load image by video name and frame index"""
        img_path = self.root_path / name_video / self.image_tmpl.format(frame_idx)
        return Image.open(img_path).convert('RGB')

    @staticmethod
    def _check_video_folders(data_root: Path, video_list):
        """Check if the video folders are not missing"""
        all_video_folders_set = {x.relative_to(data_root).name for x in data_root.iterdir()}
        required_video_folders = {video_info.name for video_info in video_list}
        missing_folders = required_video_folders - all_video_folders_set
        if missing_folders:
            raise FileNotFoundError(
                f'Missing video folders in the folder '
                f'"{data_root}": {missing_folders}'
            )

    def _check_video_frames(self, data_root: Path, video_list):
        """Check if the video frames are not missing"""

        all_jpg_files = list(data_root.glob('*/*.jpg'))
        video_files_map = {}

        for file_path in all_jpg_files:
            rel_path = file_path.relative_to(data_root)
            video_files_map.setdefault(rel_path.parent.name, set()).add(rel_path.name)

        for video_info in video_list:
            if video_info.name not in video_files_map:
                raise FileNotFoundError(f'Cannot find directory "{video_info.name}" in "{data_root}".')

            required_frames_set = {
                self.image_tmpl.format(frame_idx + 1)
                for frame_idx in range(video_info.num_frames)
            }

            detected_frames_set = video_files_map[video_info.name]

            missing_frames = required_frames_set - detected_frames_set
            if missing_frames:
                raise FileNotFoundError(
                    f'Missing frames in the folder '
                    f'"{data_root / video_info.name}": {missing_frames}'
                )

    def _parse_list(self, data_root, file_name):
        """Load markup data and check correct dataset"""

        print('Load markup data...')
        with open(file_name, 'r') as markup_file:
            video_list = [
                VideoRecord(item)
                for item in markup_file
            ]

        print('Check correct dataset...')
        if self._check_frames:
            self._check_video_frames(data_root, video_list)
        else:
            self._check_video_folders(data_root, video_list)

        print('video number:', len(video_list))
        return video_list

    def _sample_indices(self, record):
        """Sample random indices of frames"""

        average_duration = record.num_frames // self.num_segments
        if average_duration > 0:
            frames_indices = np.arange(self.num_segments) * average_duration
            random_offsets = np.random.randint(average_duration, size=self.num_segments)
            return frames_indices + random_offsets

        if record.num_frames > self.num_segments:
            return np.sort(np.random.randint(record.num_frames, size=self.num_segments))

        return np.zeros(self.num_segments)

    def _get_test_indices(self, record):
        # Picking "num_segments" frames with a (mostly) uniform step
        ns = self.num_segments
        step = record.num_frames / ns
        frames_indices = np.linspace(step / 2, step * (ns - 0.5), ns, dtype=int)
        return frames_indices

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments:
            return self._get_test_indices(record)

        return np.zeros(self.num_segments)

    def __getitem__(self, index):
        record = self.video_list[index]

        if self.test_mode:
            segment_indices = self._get_test_indices(record)
        elif self.random_shift:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_val_indices(record)

        images = [self._load_image(record.name, seg_ind + 1) for seg_ind in segment_indices]

        return self.transforms(images), record.label

    def __len__(self):
        return len(self.video_list)


class TSNDataSetWithCombinations(TSNDataSet):
    """TSNDataSet with frames combinations"""

    def __init__(
            self,
            root_path,
            list_file,
            num_segments=3,
            image_tmpl='img_{:05d}.jpg',
            transforms=None,
            random_shift=True,
            test_mode=False,
            check_frames=False,
            subsample_num=3,
            seed=0,
            rank=0,
    ):
        super().__init__(
            root_path=root_path,
            list_file=list_file,
            num_segments=num_segments,
            image_tmpl=image_tmpl,
            transforms=transforms,
            random_shift=random_shift,
            test_mode=test_mode,
            check_frames=check_frames,
        )

        self.subsample_num = subsample_num
        self.rnd = np.random.RandomState(seed + rank)
        self.sub_samples_lengths = [
            len(get_frames_combinations(self.num_segments, scale))
            for scale in range(self.num_segments, 1, -1)
        ]

    def __getitem__(self, index):
        imgs, label = super().__getitem__(index)

        scales_num = len(self.sub_samples_lengths)
        combinations = np.zeros([scales_num, self.subsample_num], np.int32)

        for i, ssl in enumerate(self.sub_samples_lengths):
            random_combinations_indices = self.rnd.permutation(ssl)[:self.subsample_num]
            combinations[i, :len(random_combinations_indices)] = random_combinations_indices

        repeats_num = imgs.shape[0] // 3 // self.num_segments
        combinations = combinations[None].repeat(repeats_num, 0)

        return imgs, combinations, label


def _check_dataroot(root_dir: Path):
    if not root_dir.is_dir():
        raise NotADirectoryError(f'The specified dataset root is not a directory: "{root_dir}"')


def get_categories_list(root_dir, categories_list_name):
    """Get the list of categories

    Args:
        root_dir: Path to the dataset root directory
        categories_list_name: Name of the file containing the dataset categories.

    Returns:
        List of categories, presented in the dataset.
    """
    root_dir = Path(root_dir).resolve()
    _check_dataroot(root_dir)

    categories_path = root_dir / categories_list_name

    with categories_path.open('r') as categories_file:
        categories = [item.rstrip() for item in categories_file]

    return categories


def get_jester_dataset_meta(root_dir, images_dir_name, files_list_name):
    """
    Prepare metadata for jester dataset
    Args:
        root_dir (Path): path to dataset folder
        images_dir_name (str): Name of directory, containing the folders with video frames.
        files_list_name (str): Name of the file, containing the videos names list

    Returns:
        tuple: Tuple of three elements, including  a categories list,
          a path to the list containing video names
          and a path to the directory, containing video frames.
    """
    root_dir = Path(root_dir).resolve()
    _check_dataroot(root_dir)

    videos_dir = root_dir / images_dir_name

    if not videos_dir.is_dir():
        raise NotADirectoryError(f'Cannot find the images directory at the location "{videos_dir}"')

    list_path = root_dir / files_list_name

    categories = get_categories_list(root_dir, 'categories.txt')

    return categories, list_path, videos_dir


def get_dataset_for_evaluation(
        dataset_root: Union[str, Path],
        images_dir_name: str,
        files_list_name: str,
        image_size: int,
        num_segments: int,
        subsample_num: int,
        seed: int,
):
    """Prepare the MindSpore dataset for evaluation"""
    categories, val_list_path, videos_dir = get_jester_dataset_meta(
        dataset_root,
        images_dir_name,
        files_list_name,
    )

    transforms = GroupCompose([
        GroupOverSample(image_size, 256),
        Stack(roll=True),
        HWC2CHW(),
        GroupNormalize(_PREPROCESS_MEAN, _PREPROCESS_STD),
    ])

    jester = TSNDataSetWithCombinations(
        root_path=videos_dir,
        list_file=val_list_path,
        num_segments=num_segments,
        transforms=transforms,
        image_tmpl='{:05d}.jpg',
        test_mode=True,
        subsample_num=subsample_num,
        seed=seed,
    )

    dataset = de.GeneratorDataset(
        jester,
        ["video", "combinations", "label"],
        shuffle=False,
        num_parallel_workers=1,
        python_multiprocessing=False,
    )
    dataset = dataset.batch(1)

    return dataset, len(categories)


def get_dataset_for_training(
        dataset_root: Union[str, Path],
        images_dir_name: str,
        files_list_name: str,
        image_size: int,
        num_segments: int,
        batch_size: int,
        subsample_num,
        seed,
        rank,
        group_size,
        train_workers: int = 8,
):
    """Prepare the MindSpore dataset for training"""
    categories, train_list_path, videos_dir = get_jester_dataset_meta(
        dataset_root,
        images_dir_name,
        files_list_name,
    )

    transforms = GroupCompose([
        GroupMultiScaleCrop(image_size, [1, .875, .75, .66]),
        GroupRandomHorizontalFlip(),
        Stack(roll=True),
        HWC2CHW(),
        GroupNormalize(_PREPROCESS_MEAN, _PREPROCESS_STD)
    ])

    if group_size > 1:
        shard_id = rank
        num_shards = group_size
    else:
        shard_id = None
        num_shards = None

    jester = TSNDataSetWithCombinations(
        root_path=videos_dir,
        list_file=train_list_path,
        num_segments=num_segments,
        transforms=transforms,
        image_tmpl='{:05d}.jpg',
        subsample_num=subsample_num,
        seed=seed,
        rank=rank,
    )

    dataset = de.GeneratorDataset(
        jester,
        ["video", "combinations", "label"],
        num_parallel_workers=train_workers,
        python_multiprocessing=train_workers > 1,
        num_shards=num_shards,
        shard_id=shard_id,
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset, len(categories)
