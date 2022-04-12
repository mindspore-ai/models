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
""" AUC """
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score


def getInfo(filename):
    info_list = []
    with open(filename) as f:
        for line in f.readlines():
            info_list.extend(list(line.split()))
    info = np.array([float(x) for x in info_list]).reshape(-1, 1)
    return info


def getAucReslut(pred_path, label_path):
    pred = getInfo(pred_path)
    label = getInfo(label_path)
    auc = roc_auc_score(label, pred)
    print(auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="infer auc")
    parser.add_argument("--pred_path", type=str, default="pred.txt", help="path of pred file")
    parser.add_argument("--label_path", type=str, default="label.txt", help="path of label file")
    args = parser.parse_args()
    getAucReslut(args.pred_path, args.label_path)
