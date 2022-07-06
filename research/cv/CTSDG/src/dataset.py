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
"""Dataset for CTSDG"""

import random
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
from mindspore.dataset import DistributedSampler
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import RepeatDataset
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.dataset.vision.py_transforms import CenterCrop
from mindspore.dataset.vision.py_transforms import Decode
from mindspore.dataset.vision.py_transforms import Grayscale
from mindspore.dataset.vision.py_transforms import Normalize
from mindspore.dataset.vision.py_transforms import Resize
from mindspore.dataset.vision.py_transforms import ToTensor
from mindspore.dataset.vision.utils import Inter
from skimage.color import rgb2gray
from skimage.feature import canny


class ImageDataset:
    """Base dataset class with images and masks for CTSDG"""
    def __init__(self, config, is_training=False):
        self.load_size = config.image_load_size
        self.sigma = config.sigma
        self.is_training = is_training

        if is_training:
            image_part_index = config.anno_train_index
            masks_root = config.train_masks_root
        else:
            image_part_index = config.anno_eval_index
            masks_root = config.eval_masks_root

        self.image_dataset = get_images_dataset(config.data_root, config.anno_path, image_part_index)
        self.masks_dataset = get_images_dataset(masks_root)

        self.number_image = len(self.image_dataset)
        self.number_mask = len(self.masks_dataset)

        self.image_transforms = Compose(
            [
                Decode(),
                CenterCrop((178, 178)),
                Resize(config.image_load_size, Inter.BILINEAR),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                np.asarray,
            ]
        )
        self.masks_transforms = Compose(
            [
                Decode(),
                Grayscale(),
                Resize(config.image_load_size, Inter.NEAREST),
                np.asarray,
            ]
        )

    def __getitem__(self, index: int):
        with open(self.image_dataset[index % self.number_image], 'rb') as f:
            image = f.read()
        image = self.image_transforms(image)[0].astype(np.float32)

        if self.is_training:
            mask_index = random.randint(0, self.number_mask - 1)
        else:
            mask_index = index % self.number_mask

        with open(self.masks_dataset[mask_index], 'rb') as f:
            mask = f.read()
        mask = self.masks_transforms(mask)[0] / 255.0

        threshold = 0.5
        ones = mask >= threshold
        mask = (1 - ones).astype(np.float32)

        edge, gray_image = self.image_to_edge(image)

        mask = np.expand_dims(mask, axis=0)
        gray_image = np.expand_dims(gray_image, axis=0)
        edge = np.expand_dims(edge, axis=0)

        return image, mask, edge, gray_image

    def __len__(self):
        return self.number_image

    def image_to_edge(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get edge and gray image of specified image

        Args:
            image: numpy image with values in [-1, 1] range

        Returns:
            Image edges and image in gray format
        """
        gray_image = rgb2gray(np.asarray(image.transpose((1, 2, 0)) * 255.0, np.uint8)).astype(np.float32)
        edge = canny(gray_image, sigma=self.sigma).astype(np.float32)

        return edge, gray_image


def get_images_dataset(data_root: str, anno_path=None, index=0) -> List[str]:
    """
    Find all images in specified dir and return all paths in list

    Args:
        data_root: Images root dir
        anno_path: Path to the annotation file with filenames and test/eval partitions
        index: Index to select

    Returns:
        List with image paths in root dir
    """
    root_dir = Path(data_root)
    images = []
    img_ext = ['.png', '.jpg', '.jpeg']
    if anno_path:
        if not Path(anno_path).exists():
            raise FileExistsError(f'Error - anno_path does not exist in this location:\n{anno_path}!')

        with open(anno_path, 'r') as f:
            anno_lines = f.readlines()

        for line in anno_lines:
            attr = line.split(' ')
            if int(attr[-1]) == index:
                img_path = root_dir / attr[0]
                if img_path.is_file() and img_path.suffix in img_ext:
                    images.append(img_path.as_posix())
    else:
        for img_path in root_dir.glob('*'):
            if img_path.is_file() and img_path.suffix in img_ext:
                images.append(img_path.as_posix())

    return sorted(images)


def create_ctsdg_dataset(config, is_training: bool) -> RepeatDataset:
    """
    Create dataset for CTSDG training

    Args:
        config: model's configuration
        is_training: prepare dataset for training or eval

    Returns:
        Mindspore dataset instance for training or eval
    """
    image_dataset = ImageDataset(config, is_training)
    column_names = ['image', 'mask', 'edge', 'gray_image']
    sampler = DistributedSampler(
        config.device_num,
        config.rank_id,
        shuffle=is_training,
    )
    ms_dataset = GeneratorDataset(
        image_dataset,
        column_names=column_names,
        num_parallel_workers=config.num_parallel_workers,
        max_rowsize=20,
        sampler=sampler,
    )
    batch_size = config.batch_size if is_training else config.test_batch_size
    ms_dataset = ms_dataset.batch(batch_size, drop_remainder=True)
    config.length_dataset = len(image_dataset) // config.batch_size
    return ms_dataset
