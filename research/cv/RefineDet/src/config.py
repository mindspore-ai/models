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
"""Config parameters for RefineDet models."""

from .config_vgg16 import config_320 as config_vgg16_320
from .config_vgg16 import config_512 as config_vgg16_512
from .config_resnet101 import config_320 as config_resnet_320
from .config_resnet101 import config_512 as config_resnet_512

config = None

config_map = {
    "refinedet_vgg16_320": config_vgg16_320,
    "refinedet_vgg16_512": config_vgg16_512,
    "refinedet_resnet101_320": config_resnet_320,
    "refinedet_resnet101_512": config_resnet_512,
}

def get_config(using_model="refinedet_vgg16_320", using_dataset="voc_test"):
    """init config according to args"""
    global config
    if config is not None:
        return config
    config = config_map[using_model]
    if using_dataset == "voc0712":
        config.voc_root = config.voc0712_root
        config.num_classes = config.voc_num_classes
        config.classes = config.voc_classes
    elif using_dataset == "voc0712plus":
        config.voc_root = config.voc0712plus_root
        config.num_classes = config.voc_num_classes
        config.classes = config.voc_classes
    elif using_dataset == "voc_test":
        config.voc_root = config.voc_test_root
        config.num_classes = config.voc_num_classes
        config.classes = config.voc_classes
    elif using_dataset == "coco":
        config.num_classes = config.coco_num_classes
        config.classes = config.coco_classes
    # calculate the boxes number
    if config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.steps[i]) * (w // config.steps[i]) * config.num_default[i]
        config.num_ssd_boxes = num
    return config
