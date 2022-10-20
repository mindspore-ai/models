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

# This file was copied from project [ZhaoWeicheng][Pyramidbox.pytorch]

import os
from easydict import EasyDict
import numpy as np


_C = EasyDict()
cfg = _C
# data argument config
_C.expand_prob = 0.5
_C.expand_max_ratio = 4
_C.hue_prob = 0.5
_C.hue_delta = 18
_C.contrast_prob = 0.5
_C.contrast_delta = 0.5
_C.saturation_prob = 0.5
_C.saturation_delta = 0.5
_C.brightness_prob = 0.5
_C.brightness_delta = 0.125
_C.data_anchor_sampling_prob = 0.5
_C.min_face_size = 6.0
_C.apply_distort = True
_C.apply_expand = True
_C.img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
_C.resize_width = 640
_C.resize_height = 640
_C.scale = 1 / 127.0
_C.anchor_sampling = True
_C.filter_min_face = True

# train config
_C.LR_STEPS = [80000, 100000, 120000]
_C.DIS_LR_STEPS = [30000, 35000, 40000]

# anchor config
_C.FEATURE_MAPS = [[160, 160], [80, 80], [40, 40], [20, 20], [10, 10], [5, 5]]
_C.INPUT_SIZE = (640, 640)
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES = [16, 32, 64, 128, 256, 512]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# loss config
_C.NUM_CLASSES = 2
_C.OVERLAP_THRESH = 0.35
_C.NEG_POS_RATIOS = 3


# detection config
_C.NMS_THRESH = 0.3
_C.TOP_K = 5000
_C.KEEP_TOP_K = 750
_C.CONF_THRESH = 0.05


# dataset config
_C.HOME = '/data2/James/dataset/pyramidbox_dataset/'

# face config
_C.FACE = EasyDict()
_C.FACE.FILE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../data'
_C.FACE.TRAIN_FILE = os.path.join(_C.FACE.FILE_DIR, 'face_train.txt')
_C.FACE.VAL_FILE = os.path.join(_C.FACE.FILE_DIR, 'face_val.txt')
_C.FACE.FDDB_DIR = os.path.join(_C.HOME, 'FDDB')
_C.FACE.WIDER_DIR = os.path.join(_C.HOME, 'WIDERFACE')
_C.FACE.OVERLAP_THRESH = 0.35
