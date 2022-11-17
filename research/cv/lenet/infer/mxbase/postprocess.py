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
import sys
import numpy as np


batch_size = 1


def calcul_acc(labels, preds):
    """
        a private function for calculating accuracy
        Args:
            labels (Object): actual labels
            preds (Object): predict labels
        Returns:
            None
    """
    return sum(1 for x, y in zip(labels, preds) if x == y) / len(labels)


def get_result(result_dir, img_dir):
    """
        a public function for getting result about accuracy

        Args:
            result_dir (Object): output path of inference
            img_dir (Object): image path of inference
        Returns:
            None
    """
    print("Start calculating accuracy,waiting...")
    files = os.listdir(img_dir)
    preds = []
    labels = []
    for f in files:
        file_name = f.split('.')[0]
        label = int(file_name.split('_')[-1])
        labels.append(label)
        output = np.fromfile(os.path.join(result_dir, file_name + '.bin'), dtype=np.float32)
        output = output[0:10]
        preds.append(np.argmax(output, axis=0))
    acc = calcul_acc(labels, preds)
    print("accuracy: {}".format(acc))
    with open('acc.log', 'w')as f:
        f.write("'Accuracy':{}".format(acc))


if __name__ == '__main__':
    result_path = sys.argv[1]
    img_path = sys.argv[2]
    get_result(result_path, img_path)
