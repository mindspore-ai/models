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
"""Preprocess images for HiFaceGAN model dataset"""
import os
import sys

import cv2
import imgaug
import imgaug.augmenters as ia
import numpy as np
from tqdm import tqdm


def get_down():
    """Downgrade augmentations"""
    return ia.Sequential([
        ia.Resize((0.125, 0.25)),
        ia.Resize({"height": 512, "width": 512}),
    ])


def get_noise():
    """Noise augmentations"""
    return ia.OneOf([
        ia.AdditiveGaussianNoise(scale=(20, 40), per_channel=True),
        ia.AdditiveLaplaceNoise(scale=(20, 40), per_channel=True),
        ia.AdditivePoissonNoise(lam=(15, 30), per_channel=True),
    ])


def get_blur():
    """Blur augmentations"""
    return ia.OneOf([
        ia.MotionBlur(k=(10, 20)),
        ia.GaussianBlur((3.0, 8.0)),
    ])


def get_jpeg():
    """JPEG compression augmentation"""
    return ia.JpegCompression(compression=(50, 85))


def get_full():
    """Full set of augmentations"""
    return ia.Sequential([
        get_blur(),
        get_noise(),
        get_jpeg(),
        get_down(),
    ], random_order=True)


def get_by_suffix(suffix='full'):
    """Get augmentations by suffix"""
    if suffix == 'down':
        return get_down()
    if suffix == 'noise':
        return get_noise()
    if suffix == 'blur':
        return get_blur()
    if suffix == 'jpeg':
        return get_jpeg()
    if suffix == 'full':
        return get_full()
    raise ValueError(f'Suffix {suffix} is not supported')


def preprocess_images(input_dir, output_dir, suffix='full'):
    """Preprocess images and write the output into the specified directory"""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f'Directory {input_dir} does not exist')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    trans = get_by_suffix(suffix)

    for item in tqdm(os.listdir(input_dir)):
        hr = cv2.imread(os.path.join(input_dir, item))

        hr = ia.Resize({"height": 512, "width": 512}).augment_image(hr)
        lr = trans.augment_image(hr)
        img = np.concatenate((lr, hr), axis=0)
        cv2.imwrite(os.path.join(output_dir, item), img)


def create_mixed_dataset(data_path, prefix, degradation_type, start_index, end_index):
    """Creates dataset with mixed augmentations"""
    if start_index >= end_index:
        raise ValueError('Start index should be lower than end_index')

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f'Directory {data_path} does not exist')

    target_dir = os.path.join(data_path, f'{prefix}_{degradation_type}')

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    for index in range(start_index, end_index + 1):
        source_dir = os.path.join(data_path, '%05d' % (index * 1000))
        preprocess_images(source_dir, target_dir, degradation_type)


def run_main():
    """Main function"""
    _, data_path, prefix, degradation_type, start_index, end_index = sys.argv
    start_index = int(start_index)
    end_index = int(end_index)
    create_mixed_dataset(data_path, prefix, degradation_type, start_index, end_index)


if __name__ == '__main__':
    imgaug.seed(0)
    run_main()
