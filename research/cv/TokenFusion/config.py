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

import numpy as np

# DATASET PARAMETERS
TRAIN_DIR = "./data/nyudv2"  # 'Modify data path'
VAL_DIR = TRAIN_DIR
TRAIN_LIST = "./data/nyudv2/train.txt"
VAL_LIST = "./data/nyudv2/val.txt"


SHORTER_SIDE = 350
CROP_SIZE = 500
RESIZE_SIZE = None

NORMALISE_PARAMS = [
    1.0 / 255,  # Image SCALE
    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # Image MEAN
    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),  # Image STD
    1.0 / 5000,
]  # Depth SCALE
IGNORE_LABEL = 255
NUM_CLASSES = 40
