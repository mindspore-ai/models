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

from mindspore import context

train_img_path = "./dataset/train_dataset/images"
train_gt_path = "./dataset/train_dataset/labels"
train_edge_path = "./dataset/train_dataset/edges"

DUT_OMRON_img_path = "./dataset/test_dataset/DUT-OMRON/DUT-OMRON-image"
DUT_OMRON_gt_path = "./dataset/test_dataset/DUT-OMRON/DUT-OMRON-mask"
DUTS_TE_img_path = "./dataset/test_dataset/DUTS-TE/DUTS-TE-Image"
DUTS_TE_gt_path = "./dataset/test_dataset/DUTS-TE/DUTS-TE-Mask"
ECCSD_img_path = "./dataset/test_dataset/ECCSD/ECCSD-image"
ECCSD_gt_path = "./dataset/test_dataset/ECCSD/ECCSD-mask"
HKU_IS_img_path = "./dataset/test_dataset/HKU-IS/HKU-IS-image"
HKU_IS_gt_path = "./dataset/test_dataset/HKU-IS/HKU-IS-mask"
SOD_img_path = "./dataset/test_dataset/SOD/SOD-image"
SOD_gt_path = "./dataset/test_dataset/SOD/SOD-mask"

batch_size = 10
train_size = 224

device_target = 'GPU'
LR = 2e-5
WD = 0.0005
EPOCH = 100

MODE = context.GRAPH_MODE
ckpt_file = "PAGENET.ckpt"

file_name = 'pagenet'
file_format = 'MINDIR'
