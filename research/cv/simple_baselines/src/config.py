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
"""Config parameters for simple baselines."""
from easydict import EasyDict as edict

config = edict()

# general
config.GENERAL = edict()
config.GENERAL.VERSION = 'commit'
config.GENERAL.TRAIN_SEED = 1
config.GENERAL.EVAL_SEED = 1
config.GENERAL.DATASET_SEED = 1
config.GENERAL.RUN_DISTRIBUTE = True

# model arts
config.MODELARTS = edict()
config.MODELARTS.IS_MODEL_ARTS = False
config.MODELARTS.CACHE_INPUT = '/cache/data_tzh/'
config.MODELARTS.CACHE_OUTPUT = '/cache/train_out/'

# model
config.MODEL = edict()
config.MODEL.IS_TRAINED = False
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = 'resnet50.ckpt'
config.MODEL.NUM_JOINTS = 17
config.MODEL.IMAGE_SIZE = [192, 256]

# network
config.NETWORK = edict()
config.NETWORK.NUM_LAYERS = 50
config.NETWORK.DECONV_WITH_BIAS = False
config.NETWORK.NUM_DECONV_LAYERS = 3
config.NETWORK.NUM_DECONV_FILTERS = [256, 256, 256]
config.NETWORK.NUM_DECONV_KERNELS = [4, 4, 4]
config.NETWORK.FINAL_CONV_KERNEL = 1
config.NETWORK.REVERSE = True
config.NETWORK.TARGET_TYPE = 'gaussian'
config.NETWORK.HEATMAP_SIZE = [48, 64]
config.NETWORK.SIGMA = 2

# loss
config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True

# dataset
config.DATASET = edict()
config.DATASET.TYPE = 'COCO'
config.DATASET.ROOT = '/home/dataset/coco/'
config.DATASET.TRAIN_SET = 'train2017'
config.DATASET.TRAIN_JSON = 'annotations/person_keypoints_train2017.json'
config.DATASET.TEST_SET = 'val2017'
config.DATASET.TEST_JSON = 'annotations/person_keypoints_val2017.json'

# training data augmentation
config.DATASET.FLIP = True
config.DATASET.SCALE_FACTOR = 0.3
config.DATASET.ROT_FACTOR = 40

# train
config.TRAIN = edict()
config.TRAIN.SHUFFLE = True
config.TRAIN.BATCH_SIZE = 64  # 32 in paper
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140
config.TRAIN.LR = 0.001
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 120]
config.TRAIN.NUM_PARALLEL_WORKERS = 8
config.TRAIN.SAVE_CKPT = True
config.TRAIN.CKPT_PATH = "/home/model/"
config.TRAIN.SAVE_CKPT_EPOCH = 3
config.TRAIN.KEEP_CKPT_MAX = 10

# valid
config.TEST = edict()
config.TEST.BATCH_SIZE = 1
config.TEST.FLIP_TEST = True
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True
config.TEST.USE_GT_BBOX = False
config.TEST.NUM_PARALLEL_WORKERS = 8
config.TEST.MODEL_FILE = '/home/model/multi_train_poseresnet_commit_5-140_292.ckpt'
config.TEST.COCO_BBOX_FILE = '/home/dataset/coco/annotations/COCO_val2017_detections_AP_H_56_person.json'
config.TEST.OUTPUT_DIR = '/home/results/'

# nms
config.TEST.OKS_THRE = 0.9
config.TEST.IN_VIS_THRE = 0.2
config.TEST.BBOX_THRE = 1.0
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0

# 310 infer-related
config.INFER = edict()
config.INFER.PRE_RESULT_PATH = './preprocess_Result'
config.INFER.POST_RESULT_PATH = './result_Files'

# Help description for each configuration
config.enable_modelarts = "Whether training on modelarts, default: False"
config.data_url = "Url for modelarts"
config.train_url = "Url for modelarts"
config.data_path = "The location of the input data."
config.output_path = "The location of the output file."
config.device_target = "Running platform, choose from Ascend, GPU or CPU, and default is Ascend."
config.enable_profiling = 'Whether enable profiling while training, default: False'
# Parameters that can be modified at the terminal
config.ckpt_save_dir = "ckpt path to save"
config.batch_size = "training batch size"
config.run_distribute = "Run distribute, default is false."
