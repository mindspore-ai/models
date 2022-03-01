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
############## Count accuracy by bin files #################
python count_acc.py
"""


import os
import argparse

import numpy as np

PARSER = argparse.ArgumentParser(description="ISyNet accuracy counter")
PARSER.add_argument("--gt_path", type=str, required=True, help="path with ground truth files")
PARSER.add_argument("--predict_path", type=str, required=True, help="path with prediction files")
ARGS = PARSER.parse_args()

if __name__ == '__main__':
    results = []
    for file in os.listdir(ARGS.gt_path):
        if file.endswith(".bin"):
            index = file.split("_")[1]
            gt_label = np.fromfile(os.path.join(ARGS.gt_path, file), np.int32)
            pred_file = os.path.join(ARGS.predict_path, f"imagenet_{index}", "output_0.bin")
            pred_label = np.fromfile(pred_file, np.float32).argmax()
            results.append(gt_label == pred_label)
    print("Model accuracy: ", np.mean(results))
