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

"""
This file contains definitions of useful data structures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

# Define paths to each dataset
LSP_ROOT = '/XXXX/XXXX/LSP/lsp_dataset'
LSP_ORIGINAL_ROOT = '/XXXX/XXXX/LSP/lsp_dataset_original'
UPI_S1H_ROOT = '/XXXX/XXXX/UP_3D/upi-s1h'
MPII_ROOT = '/XXXX/XXXX/MPII'
COCO_ROOT = '/XXXX/XXXX/coco2014'
UP_3D_ROOT = '/XXXX/XXXX/UP_3D/up-3d'

# CKPT file path of ResNet-50 model
PRETRAINED_RESNET_50 = 'pretrained_resnet50.ckpt'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'datasets'

# Path to test/train npz files
DATASET_FILES = [{'lsp': join(DATASET_NPZ_PATH, 'extras', 'lsp_dataset_test.npz'),
                  'up-3d': join(DATASET_NPZ_PATH, 'extras', 'up_3d_lsp_test.npz')
                  },
                 {'lsp-orig': join(DATASET_NPZ_PATH, 'extras', 'lsp_dataset_original_train.npz'),
                  'up-3d': join(DATASET_NPZ_PATH, 'extras', 'up_3d_trainval.npz'),
                  'mpii': join(DATASET_NPZ_PATH, 'extras', 'mpii_train.npz'),
                  'coco': join(DATASET_NPZ_PATH, 'extras', 'coco_2014_train.npz'),
                  }
                ]

# Original dataset folders to generate test/train npz files
DATASET_FOLDERS = {'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                   'up-3d': UP_3D_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT
                   }

SMPL_CKPT_FILE = 'data/smpl.ckpt'
MESH_CKPT_FILE = 'data/mesh.ckpt'

"""
Each dataset uses different sets of joints.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are:
0 - Right Ankle
1 - Right Knee
2 - Right Hip
3 - Left Hip
4 - Left Knee
5 - Left Ankle
6 - Right Wrist
7 - Right Elbow
8 - Right Shoulder
9 - Left Shoulder
10 - Left Elbow
11 - Left Wrist
12 - Neck (LSP definition)
13 - Top of Head (LSP definition)
14 - Pelvis (MPII definition)
15 - Thorax (MPII definition)
16 - Spine (Human3.6M definition)
17 - Jaw (Human3.6M definition)
18 - Head (Human3.6M definition)
19 - Nose
20 - Left Eye
21 - Right Eye
22 - Left Ear
23 - Right Ear
"""

# In SMPL
PARENT = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
