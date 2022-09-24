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

import os

# Path of original dataset
DATASET_DIR = "XXXXXXXXXXXXXX/dataset"

# Path of FDDB dataset
FDDB_DIR = os.path.join(DATASET_DIR, 'FDDB')

# Store train data
TRAIN_DATA_DIR = os.path.join(DATASET_DIR, "train_data")

# Path of mindrecords
MINDRECORD_DIR = os.path.dirname(os.path.realpath(__file__)) + "/mindrecords"

# Path of checkpoints
CKPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints"

# Configure the ratio of each loss
RADIO_CLS_LOSS = 1.0
RADIO_BOX_LOSS = 0.5
RADIO_LANDMARK_LOSS = 0.5

# Path to store logs
LOG_DIR = os.path.dirname(os.path.realpath(__file__)) + "/log"

TRAIN_BATCH_SIZE = 384

TRAIN_LR = 0.001

END_EPOCH = 30

MIN_FACE_SIZE = 20
SCALE_FACTOR = 0.79

P_THRESH = 0.6
R_THRESH = 0.7
O_THRESH = 0.7
