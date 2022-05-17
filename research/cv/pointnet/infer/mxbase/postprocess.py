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
"""postprocess"""
import os
import glob
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default="../datapath_BS1/", help='dataset_path')
parser.add_argument('--result_path', default="./mxbase/result", help='result_path')
parser.add_argument('--num_classes', type=int,
                    help='num_classes', default=4)
args = parser.parse_args()
dataset_path = args.dataset_path
result_path = args.result_path
num_classes = args.num_classes
shape_ious = []
label_path = os.path.join(dataset_path, 'labels_ids.npy')
file_list = []
file_list1 = glob.glob(result_path+'/*')
for i in range(len(file_list1)):
    file_list.append(result_path+'/shapenet_data_bs1_%03d'%i+'.bin')
#print(file_list)
for i, file_name in enumerate(file_list):
    print(file_name)
    data = np.fromfile(file_name, dtype=np.float32)
    label = np.load(label_path)
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
