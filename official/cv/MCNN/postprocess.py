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
"""postprocess for 310 inference"""
import os
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description="postprocess")
label_path = "../test_data/ground_truth_csv/"
parser.add_argument("--result_dir", type=str, default="./ascend310_infer/result_Files", help="result files path.")
parser.add_argument("--label_dir", type=str, default=label_path, help="image file path.")
args = parser.parse_args()


if __name__ == '__main__':

    rst_path = args.result_dir
    # labels = np.load(args.label_dir, allow_pickle=True)
    label_files = [filename for filename in os.listdir(args.label_dir) \
                  if os.path.isfile(os.path.join(args.label_dir, filename))]
    label_files.sort()
    mae = 0
    mse = 0
    for idx, label_file in enumerate(label_files):
        # os.path.join(label_path, "IMG_"+str(idx+1)+".csv")
        den = pd.read_csv(os.path.join(label_path, "IMG_"+str(idx+1)+".csv"), sep=',', header=None).values
        den = den.astype(np.float32, copy=False)

        f_name = os.path.join(rst_path, "IMG" + "_" + str(idx+1) + "_0.bin")
        pred = np.fromfile(f_name, np.float32)
        gt_count = np.sum(den)
        et_count = np.sum(pred)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
        print(os.path.join(label_path, "IMG_"+str(idx+1)+".csv"), np.sum(den))
        print(f_name, np.sum(pred))
    mae = mae / 182
    mse = np.sqrt(mse / 182)
    print('MAE:', mae, '  MSE:', mse)
