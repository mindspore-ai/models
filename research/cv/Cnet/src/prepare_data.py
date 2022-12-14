# Copyright 2021 Huawei Technologies Co., Ltd
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
# This file was refer to proiect https://pytorch.org/vision/0.8/_modules/torchvision/datasets/phototour.html#PhotoTour
"""read ubcdata"""
import os
from typing import List
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import save_checkpoint
from PIL import Image


def _check_path_exists(path) -> bool:
    return os.path.exists(path)


def read_image_file(data_dir: str, image_ext: str, n: int) -> np.array:
    """Return a Tensor containing the patches."""

    def PIL2array(_img: Image.Image) -> np.ndarray:
        """Convert PIL image type to numpy 2D array."""
        return np.array(_img.getdata(), dtype=np.uint8).reshape(64, 64)

    def find_files(_data_dir: str, _image_ext: str) -> List[str]:
        """Return a list with the file names of the images containing the
        patches."""
        files = []
        # find those files with the specified extension
        for file_dir in os.listdir(_data_dir):
            if file_dir.endswith(_image_ext):
                files.append(os.path.join(_data_dir, file_dir))
        return sorted(files)  # sort files in ascend order to keep relations

    patches = []
    list_files = find_files(data_dir, image_ext)

    for fpath in list_files:
        img = Image.open(fpath)
        for y in range(0, 1024, 64):
            for x in range(0, 1024, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                patches.append(PIL2array(patch))
    return Tensor(patches[:n], mstype.int8)


def read_info_file(data_dir: str, info_file: str) -> np.array:
    """Return a Tensor containing the list of labels Read the file and keep
    only the ID of the 3D point."""
    with open(os.path.join(data_dir, info_file), 'r') as f:
        labels = [int(line.split()[0]) for line in f]
    return Tensor(labels, mstype.int64)


def read_matches_files(data_dir: str, matches_file: str) -> np.array:
    """Return a Tensor containing the ground truth matches Read the file and
    keep only 3D point ID.

    Matches are represented with a 1, non matches with a 0.
    """
    matches = []
    with open(os.path.join(data_dir, matches_file), 'r') as f:
        for line in f:
            line_split = line.split()
            matches.append([
                int(line_split[0]),
                int(line_split[3]),
                int(line_split[1] == line_split[4])
            ])
    return Tensor(matches, mstype.int64)


def cache_data(data_dir: str, data_file: str, img_ext: str, lens: int,
               info_file: str, matches_files: str):
    print('# Caching data: {}...\n'.format(data_file.split('/')[-1]))

    data_dict = {}
    data_dict['name'] = 'data'
    data_dict['data'] = read_image_file(data_dir, img_ext, lens)

    labels_dict = {}
    labels_dict['name'] = 'labels'
    labels_dict['data'] = read_info_file(data_dir, info_file)

    matches_dict = {}
    matches_dict['name'] = 'matches'
    matches_dict['data'] = read_info_file(data_dir, matches_files)

    save_list = [data_dict, labels_dict, matches_dict]
    save_checkpoint(save_list, data_file)

    print('# Cached data successfully!\n')


def prepare_data(root: str,
                 name: str,) -> None:
    lens = {
        'notredame': 468159,
        'yosemite': 633587,
        'liberty': 450092,
    }
    image_ext = 'bmp'
    info_file = 'info.txt'
    matches_files = 'm50_100000_100000_0.txt'

    root = os.path.realpath(root)
    if not _check_path_exists(root):
        raise FileExistsError(
            '{}: the path is wrong or not exist'.format(root))

    data_file = os.path.join(root, '{}.ckpt'.format(name))
    if _check_path_exists(data_file):
        print('{}.ckpt already exists.'.format(name))
        return

    data_dir = os.path.join(root, name)
    if not _check_path_exists(data_dir):
        raise FileExistsError(
            '{}: the path is wrong or the utils does not exist.'.format(
                data_dir))

    cache_data(data_dir, data_file, image_ext, lens[name], info_file,
               matches_files)
