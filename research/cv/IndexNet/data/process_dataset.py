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
"""Dataset preparation script."""
import math
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np

from src.cfg.config import config

# Original dataset without splitting into train/test parts.
# We manually selected 50 test images and fixed its names to be able to reproduce results.
_AIM500_VALIDATION_SUBSET = (
    '247d445d', '0530e2e7', '8431ecb9', '76c7dc77', 'bab88684',
    'fabaf6e3', 'bde965af', 'e4a72c06', 'a974d3a3', 'e92f575c',
    'dc288b1a', 'e30ce6cb', '7c642e86', '57e4d780', '52a2115b',
    '33a4da38', '77d7a529', '2be0f6c9', '0a5e5a64', 'b71d875b',
    'b4157744', '0723d28f', 'a3798f05', '404618b5', '780f404a',
    '8e2eb72f', '7c88f64e', '6e470e4a', '819ee421', '12c828bc',
    'b2dfc20d', '80193d44', '51873814', 'b40d228a', '26268b4b',
    '4729fa87', 'fe6f4047', 'e92b90fc', '23bbcc9d', '197d7af1',
    '868c53f0', '7b1db264', '4852f7d4', 'd2a1bff3', '751bcc44',
    '44dd34a4', '1ea2b894', '9992c618', 'e62df10b', 'dbef692f',
)


def listdir(folder, file_format=".jpg"):
    """
    Search files into chosen dir.

    Args:
        folder (pathlib.Path): Path to folder.
        file_format (str): Search files of current format.

    Returns:
        out (list): Names of files found into folder.
    """
    return [file.name for file in folder.iterdir() if file.name.endswith(f'{file_format}')]


def set_data_structure(dataset_path):
    """
    Train test split and organize dataset structure.

    Args:
        dataset_path (pathlib.Path): Path to main dataset folder.
    """
    test_names = ['o_' + file + '.jpg' for file in _AIM500_VALIDATION_SUBSET]

    data_dirs = ['validation/original', 'validation/mask', 'validation/trimap', 'train']
    for data_dir in data_dirs:
        Path(dataset_path, data_dir).mkdir(parents=True)

    for name in test_names:
        shutil.move(f"{dataset_path}/original/{name}", f"{dataset_path}/validation/original/{name}")
        name = name.replace('jpg', 'png')
        shutil.move(f"{dataset_path}/mask/{name}", f"{dataset_path}/validation/mask/{name}")
        shutil.move(f"{dataset_path}/trimap/{name}", f"{dataset_path}/validation/trimap/{name}")

    shutil.move(f"{dataset_path}/original", f"{dataset_path}/train/original")
    shutil.move(f"{dataset_path}/mask", f"{dataset_path}/train/mask")
    shutil.rmtree(f"{dataset_path}/trimap")


def composite4(fg, bg, a, w, h):
    """
    Place foreground over background by mask.

    Args:
        fg (np.array): Foreground image.
        bg (np.array): Background image.
        a (np.array): Mask of the foreground image.
        w (int): Width of the foreground image.
        h (int): Height of the foreground image.

    Returns:
        comp (np.array): Foreground placed by mask over background.
    """
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp


def process(start, num_bgs, data_root, bg_path, part="train"):
    """
    Compose foregrounds and backgrounds by mask to make dataset.

    Args:
        start (int): Start index of the background images.
        num_bgs (int): Number of unique backgrounds per one image from matting dataset.
        data_root (pathlib.Path): Path to matting dataset directory.
        bg_path (pathlib.Path): Path to backgrounds dataset directory.
        part (str): Which part of dataset to prepare (subfolder).

    Returns:
        data_part_size (int): Number of processed images at the current part of dataset.
    """
    print(f'Start processing {part} part ...')

    a_path = data_root / part / 'mask'
    out_path = data_root / part / 'merged'
    fg_dir = data_root / part / 'original'
    bg_dir = bg_path

    out_path.mkdir(parents=True, exist_ok=True)

    fg_files = listdir(fg_dir, file_format='.jpg')

    data_part_size = int(len(fg_files) * num_bgs)

    bg_files = listdir(bg_dir, file_format='.jpg')[start:start + data_part_size]

    fg_files.sort()
    bg_files.sort()

    data_schema = []
    bg_iter = iter(bg_files)
    for k, im_name in enumerate(fg_files):
        im = cv.imread(str(fg_dir / im_name))
        a = cv.imread(str(a_path / im_name.replace('jpg', 'png')), 0)
        h, w = im.shape[:2]

        bcount = 0
        for _ in range(num_bgs):
            bg_name = next(bg_iter)
            bg = cv.imread(str(bg_dir / bg_name))
            bh, bw = bg.shape[:2]
            wratio = float(w) / float(bw)
            hratio = float(h) / float(bh)
            ratio = wratio if wratio > hratio else hratio
            if ratio > 1:
                bg = cv.resize(bg, (math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

            out = composite4(im, bg, a, w, h)
            filename = Path(out_path, im_name[:len(im_name) - 4] + '_' + str(bcount) + '.png')

            cv.imwrite(str(filename), out, [cv.IMWRITE_PNG_COMPRESSION, 9])

            if part == 'train':
                # [processed, mask, foreground, background]
                data_schema_cell = [
                    str(Path(part, 'merged', filename.name)),
                    str(Path(part, 'mask', im_name.replace('.jpg', '.png'))),
                    str(Path(part, 'original', im_name)),
                    str(Path(bg_name)) + '\n',
                ]
            else:
                # [processed, mask, trimap]
                data_schema_cell = [
                    str(Path(part, 'merged', filename.name)),
                    str(Path(part, 'mask', im_name.replace('.jpg', '.png'))),
                    str(Path(part, 'trimap', im_name.replace('.jpg', '.png'))) + '\n',
                ]
            data_schema.append('|'.join(data_schema_cell))

            bcount += 1
            print(f'{k * num_bgs + bcount}/{data_part_size}')

    with Path(Path(out_path).resolve().parent, 'data.txt').open('w') as file:
        file.writelines(data_schema)

    print(f'Successfully end {part} part processing.')
    return data_part_size


if __name__ == "__main__":
    main_dataset = Path(config.data_dir)
    backgrounds = Path(config.bg_dir)

    if not main_dataset.is_dir():
        raise NotADirectoryError(f'Not valid path to main dataset: {main_dataset}')
    if not backgrounds.is_dir():
        raise NotADirectoryError(f'Not valid path to backgrounds: {backgrounds}')

    set_data_structure(dataset_path=main_dataset)

    end = process(start=0, num_bgs=config.num_bgs_train, data_root=main_dataset, bg_path=backgrounds, part='train')
    _ = process(start=end, num_bgs=config.num_bgs_val, data_root=main_dataset, bg_path=backgrounds, part='validation')
