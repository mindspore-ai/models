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

"""task metric"""

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="ternarybert task metric")
parser.add_argument("--label_path", type=str, required=True, help="label directory")
parser.add_argument("--result_path", type=str, required=True, help="result directory")
args = parser.parse_args()

BATCH_SIZE = 32
LABEL_NUM = 1

class Pearsonr:
    """Pearsonr"""
    def __init__(self):
        self.logits_array = np.array([])
        self.labels_array = np.array([])
        self.name = 'Pearsonr'

    def update(self, logits, labels):
        label = np.reshape(labels, -1)
        logit = np.reshape(logits, -1)
        self.labels_array = np.concatenate([self.labels_array, label])
        self.logits_array = np.concatenate([self.logits_array, logit])

    def get_metrics(self):
        if len(self.labels_array) < 2:
            return 0.0
        x_mean = self.logits_array.mean()
        y_mean = self.labels_array.mean()
        xm = self.logits_array - x_mean
        ym = self.labels_array - y_mean
        norm_xm = np.linalg.norm(xm)
        norm_ym = np.linalg.norm(ym)
        return np.dot(xm / norm_xm, ym / norm_ym) * 100.0


if __name__ == '__main__':
    label_numpys = np.load(args.label_path)
    callback = Pearsonr()
    file_num = len(os.listdir(args.result_path))
    for i in range(file_num):
        f_name = "tinybert_bs" + str(BATCH_SIZE) + "_" + str(i) + ".bin"
        result_numpy = np.fromfile(os.path.join(args.result_path, f_name), np.float32)
        print(f_name)
        print(result_numpy)
        label_numpy = label_numpys[i]
        callback.update(result_numpy, label_numpy)
    metrics = callback.get_metrics()
    print('{}: {}'.format(callback.name, metrics))
