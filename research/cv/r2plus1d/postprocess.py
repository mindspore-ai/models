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
'''postprocess'''
import os
import argparse

import numpy as np

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="", help="root path of predicted images")
    args_opt = parser.parse_args()
    return args_opt

def main():
    args = parse_args()
    predict_num = 0
    for file in sorted(os.listdir(args.result_path)):
        label = int(file.split("_")[0])
        y_predict = np.fromfile(os.path.join(args.result_path, file), dtype=np.float32)
        y_predict = y_predict.reshape(101)
        y_predict = np.exp(y_predict) / np.sum(np.exp(y_predict), axis=0)
        predict = np.argmax(y_predict)
        predict_num += 1 if predict == label else 0
    print("Accuracy:", predict_num/len(os.listdir(args.result_path)))
if __name__ == '__main__':
    main()
