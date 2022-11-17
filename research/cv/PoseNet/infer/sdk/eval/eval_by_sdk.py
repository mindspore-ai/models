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

args = parser.parse_args()

def get_result(result_path):
    results = np.zeros((343, 2))
    txt_file = "../dataset_test.txt"
    step = 0
    i = 0
    with open(txt_file, 'r') as f:
        next(f)
        next(f)
        next(f)
        for line in f:
            fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
            fname = str(fname)
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            image_poses = []
            image_poses.append((p0, p1, p2, p3, p4, p5, p6))
            poses = np.array(image_poses)
            pose_x = np.squeeze(poses[:, 0:3])
            pose_q = np.squeeze(poses[:, 3:])
            file_name = "posenet" + "_" + str(i)
            i = i + 1
            result_path_0 = os.path.join(result_path, file_name + "_0.bin")
            result_path_1 = os.path.join(result_path, file_name + "_1.bin")
            result_path_2 = os.path.join(result_path, file_name + "_2.bin")
            result_path_3 = os.path.join(result_path, file_name + "_3.bin")
            result_path_4 = os.path.join(result_path, file_name + "_4.bin")
            result_path_5 = os.path.join(result_path, file_name + "_5.bin")
            p1_x = np.fromfile(result_path_0, dtype=np.float32).reshape(1, 3)
            p1_q = np.fromfile(result_path_1, dtype=np.float32).reshape(1, 4)
            p2_x = np.fromfile(result_path_2, dtype=np.float32).reshape(1, 3)
            p2_q = np.fromfile(result_path_3, dtype=np.float32).reshape(1, 4)
            p3_x = np.fromfile(result_path_4, dtype=np.float32).reshape(1, 3)
            p3_q = np.fromfile(result_path_5, dtype=np.float32).reshape(1, 4)
            print(p1_x)
            print(p1_q)
            print(p2_x)
            print(p2_q)
            print(p3_x)
            print(p3_q)
            predicted_x = p3_x
            predicted_q = p3_q
            q1 = pose_q / np.linalg.norm(pose_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1, q2)))

            a = 2
            b = 180
            theta = a * np.arccos(d) * b / math.pi
            error_x = np.linalg.norm(pose_x - predicted_x)
            results[step, :] = [error_x, theta]
            step = step + 1
            print('Iteration:  ', step, ', Error XYZ (m):  ', error_x, ', Error Q (degrees):  ', theta)

    median_result = np.median(results, axis=0)
    print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')

if __name__ == '__main__':
    get_result(args.result_path)
