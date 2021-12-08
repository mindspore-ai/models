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
import json
import argparse
import numpy as np



parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--result_dir", type=str, required=True, help="result files path.")
parser.add_argument("--label_dir", type=str, required=True, help="image file path.")
args = parser.parse_args()



if __name__ == '__main__':
    batch_size = 1
    rst_path = args.result_dir
    file_list = os.listdir(rst_path)
    with open(args.label_dir, "r") as label:
        labels = json.load(label)
    success_num = 0.0
    total_num = 0.0
    acc = 0.0
    for f in file_list:
        if f.find("_1.bin") != -1:
            label = f.split("_1.bin")[0] + ".jpg"
            scrutinizer_out = np.fromfile(os.path.join(rst_path, f), np.float32)
            scrutinizer_out = scrutinizer_out.reshape(batch_size, 200)
            pred = np.argmax(scrutinizer_out, axis=1)[0]
            print("pred: ", pred)
            print("labels[label]: ", labels[label])
            total_num = total_num + 1
            if pred == labels[label]:
                success_num = success_num + 1
    acc = success_num / total_num
    print("success_num: ", success_num)
    print("total_num: ", total_num)
    print("acc: ", acc)
