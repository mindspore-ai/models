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
"""postprocess for 310 inference"""
import glob
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="lenet preprocess data")
parser.add_argument("--result_path", type=str, required=True, help="result path.")
parser.add_argument("--label_path", type=str, required=True, help="label path.")
args = parser.parse_args()
num_classes = 4
shape_ious = []
file_list = []
file_list1 = glob.glob(args.result_path+'/*')
for i in range(len(file_list1)):
    file_list.append(args.result_path+'/shapenet_data_bs1_%03d'%i+'_0.bin')
for i, file_name in enumerate(file_list):
    print("calaccuracy of ", file_name)
    data = np.fromfile(file_name, dtype=np.float32)
    label = np.load(args.label_path)
    label = label[i]
    start_time = time.time()
    pred = data.reshape(1, 2500, -1)
    pred_np = np.argmax(pred, axis=2)
    target_np = label - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
        print('='*50)
        print(np.mean(part_ious))
        print('='*50)

print("Final Miou: {}".format(np.mean(shape_ious)))
