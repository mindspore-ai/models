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
postprocess.
"""
import argparse

import numpy as np


def Accuracy(label, mask, preds):
    """Accuracy with masking."""
    preds = preds.astype(np.float32)
    correct_prediction = np.equal(np.argmax(preds, axis=1), np.argmax(label, axis=1))
    accuracy_all = correct_prediction.astype(np.float32)
    mask = mask.astype(np.float32)
    mask_reduce = np.mean(mask)
    mask = mask / mask_reduce
    accuracy_all *= mask
    return np.mean(accuracy_all)


def get_acc():
    """get infer Accuracy."""
    parser = argparse.ArgumentParser(description='postprocess')
    parser.add_argument('--dataset_name', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'], help='dataset name')
    parser.add_argument('--label_path', type=str, default='', help='y_test npy Files')
    parser.add_argument('--mask_path', type=str, default='', help='test_mask npy Files')
    args_opt = parser.parse_args()

    label_onehot = np.load(args_opt.label_path)
    test_mask = np.load(args_opt.mask_path)

    if args_opt.dataset_name == 'cora':
        pred = np.loadtxt('pred1.txt')
        pred = pred.reshape(2708, 7)
    if args_opt.dataset_name == 'citeseer':
        pred = np.loadtxt('pred2.txt')
        pred = pred.reshape(3327, 6)
    if args_opt.dataset_name == 'pubmed':
        pred = np.loadtxt('pred3.txt')
        pred = pred.reshape(19717, 3)

    acc = Accuracy(label_onehot, test_mask, pred)
    print("Test set results:", "accuracy=", "{:.5f}".format(acc))


if __name__ == '__main__':
    get_acc()
    