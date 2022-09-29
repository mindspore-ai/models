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
""" unique configs """


class Config:
    """
    Config setup
    """
    # dataset related
    class DATA:
        SAMPLING_RATE = 2
        NUM_FRAMES = 32
        MEAN = [0.45, 0.45, 0.45]
        STD = [0.225, 0.225, 0.225]
        RANDOM_FLIP = True
        TEST_CROP_SIZE = 224
        TEST_SCALE_HEIGHT = 256
        TEST_SCALE_WIDTH = 384
        MAX_NUM_BOXES_PER_FRAME = 28
        REVERSE_INPUT_CHANNEL = False

    class MODEL:
        NUM_CLASSES = 80
        ARCH = "slowfast"
        SINGLE_PATHWAY_ARCH = ["2d", "c2d", "i3d", "slow", "x3d", "mvit"]
        MULTI_PATHWAY_ARCH = ["slowfast"]

    class AVA:
        BGR = False
        TEST_FORCE_FLIP = False
        TEST_LISTS = ["val.csv"]
        DATADIR = "../data/input"
        ANN_DIR = "/ava/ava_annotations"
        FRA_DIR = "/ava/frames"

        FRAME_LIST_DIR = DATADIR + ANN_DIR
        ANNOTATION_DIR = DATADIR + ANN_DIR
        FRAME_DIR = DATADIR + FRA_DIR
        TEST_PREDICT_BOX_LISTS = [
            "person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv"]
        EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"
        LABEL_MAP_FILE = "person_box_67091280_iou90/ava_action_list_v2.1_for_activitynet_2018.pbtxt"
        GROUNDTRUTH_FILE = "ava_val_v2.2.csv"
        IMG_PROC_BACKEND = "cv2"
        DETECTION_SCORE_THRESH = 0.8
        FULL_TEST_ON_VAL = False

    class SLOWFAST:
        ALPHA = 4

    OUTPUT_DIR = "./"

    BATCH_SIZE = 1
    LOG_PERIOD = 1

    sdk_pipeline_name = b"im_slowfast"
