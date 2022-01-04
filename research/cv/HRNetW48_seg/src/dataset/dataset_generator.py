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
"""Dataset generators."""
import os

import mindspore.dataset.engine as de

from src.dataset.cityscapes import Cityscapes
from src.dataset.lip import LIP


def create_seg_dataset(data_name, data_path=None, batchsize=4, run_distribute=False, is_train=True, raw=False):
    """
    Create dataset loader.

    Args:
        data_name (str): name of dataset.
        data_path (str): dataset path.
        batchsize (int): batch size.
        run_distribute (bool): distribute training or not.
        is_train (bool): set `True` while training, otherwise `False`.
        raw (bool): if set it `False`, return mindspore dateset engine,
                    otherwise return python generator.

    Returns:
        dataset: dataset loader.
        crop_size: model input size.
        num_classes: number of classes.
        class_weights: a list of weights for each class.
    """
    if data_name == "cityscapes":
        num_classes = 19
        if is_train:
            multi_scale = True
            flip = True
            crop_size = (512, 1024)
        else:
            multi_scale = False
            flip = False
            crop_size = (1024, 2048)
        if data_path is None:
            return crop_size, num_classes
        dataset = Cityscapes(data_path,
                             num_samples=None,
                             num_classes=19,
                             multi_scale=multi_scale,
                             flip=flip,
                             ignore_label=255,
                             base_size=2048,
                             crop_size=crop_size,
                             downsample_rate=1,
                             scale_factor=16,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225],
                             is_train=is_train)
    elif data_name == "lip":
        num_classes = 20
        crop_size = (473, 473)
        if is_train:
            multi_scale = True
            flip = True
        else:
            multi_scale = False
            flip = False
        if data_path is None:
            return crop_size, num_classes
        dataset = LIP(data_path,
                      num_samples=None,
                      num_classes=20,
                      multi_scale=multi_scale,
                      flip=flip,
                      ignore_label=255,
                      base_size=473,
                      crop_size=crop_size,
                      downsample_rate=1,
                      scale_factor=11,
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      is_train=is_train)
    else:
        raise ValueError("Unsupported dataset.")
    class_weights = dataset.class_weights
    if raw:
        return dataset, crop_size, num_classes, class_weights
    if run_distribute:
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = int(os.getenv("RANK_SIZE"))
        dataset = de.GeneratorDataset(dataset, column_names=["image", "label"],
                                      num_parallel_workers=8,
                                      shuffle=True,
                                      num_shards=device_num, shard_id=device_id)
    else:
        dataset = de.GeneratorDataset(dataset, column_names=["image", "label"],
                                      num_parallel_workers=8,
                                      shuffle=True)
    dataset = dataset.batch(batchsize, drop_remainder=True)

    return dataset, crop_size, num_classes, class_weights
