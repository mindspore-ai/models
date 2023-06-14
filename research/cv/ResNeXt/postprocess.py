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
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="resnet inference")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--label_path", type=str, required=True, help="image file path.")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], help="MindIR device target.")
args = parser.parse_args()

batch_size = 1
num_classes = 1000


def get_result(result_path, label_path):
    files = os.listdir(result_path)
    with open(label_path, "r") as label:
        labels = json.load(label)

    if args.device_target == "Ascend":
        output_dtype = np.float16
    else:
        output_dtype = np.float32

    top1 = 0
    top5 = 0
    total_data = len(files)
    for file in files:
        img_ids_name = file.split("_0.")[0]
        data_path = os.path.join(result_path, img_ids_name + "_0.bin")
        result = np.fromfile(data_path, dtype=output_dtype).reshape(batch_size, num_classes)
        for batch in range(batch_size):
            predict = np.argsort(-result[batch], axis=-1)
            if labels[img_ids_name + ".JPEG"] == predict[0]:
                top1 += 1
            if labels[img_ids_name + ".JPEG"] in predict[:5]:
                top5 += 1
    print(f"Total data: {total_data}, top1 accuracy: {top1/total_data}, top5 accuracy: {top5/total_data}.")


if __name__ == "__main__":
    get_result(args.result_path, args.label_path)
