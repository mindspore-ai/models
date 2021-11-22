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
"""post process for 310 inference"""
import os
import argparse
import math
import numpy as np

parser = argparse.ArgumentParser(description='postprocess for posenet')
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--label_path", type=str, required=True, help="label file")
args = parser.parse_args()

def get_result(result_path, label_path):
    files = os.listdir(label_path)
    step = 0
    num = 0
    for file in files:
        num = num + 1
    results = np.zeros((num, 2))
    for file in files:
        file_name = file.split('.')[0]
        img_pose_path = os.path.join(label_path, file_name + ".bin")
        poses = np.fromfile(img_pose_path, dtype=np.float32).reshape(1, 7)
        pose_x = np.squeeze(poses[:, 0:3])
        pose_q = np.squeeze(poses[:, 3:])

        result_path_4 = os.path.join(result_path, file_name + "_4.bin")
        result_path_5 = os.path.join(result_path, file_name + "_5.bin")

        p3_x = np.fromfile(result_path_4, dtype=np.float32).reshape(1, 3)
        p3_q = np.fromfile(result_path_5, dtype=np.float32).reshape(1, 4)
        predicted_x = p3_x
        predicted_q = p3_q
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1, q2)))
        theta = 2 * np.arccos(d) * 180 / math.pi
        error_x = np.linalg.norm(pose_x - predicted_x)
        results[step, :] = [error_x, theta]
        print('Iteration:  ', step, ', Error XYZ (m):  ', error_x, ', Error Q (degrees):  ', theta)
        step = step + 1
    median_result = np.median(results, axis=0)
    print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')

if __name__ == '__main__':
    get_result(args.result_path, args.label_path)
