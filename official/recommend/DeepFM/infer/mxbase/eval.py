# Copyright(C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""model evaluation"""
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--pred_file',
    default='../data/preds.txt',
    help='preditions')
parser.add_argument(
    '--label_file',
    default='../data/label.txt',
    help='ground truth')

args = parser.parse_args()
print(f"Loading pred_file: {args.pred_file}")
preds = np.genfromtxt(args.pred_file, delimiter=',')
preds = np.delete(preds, np.where(np.isnan(preds))[0])

print(f"Loading label_file: {args.label_file}")
labels = np.genfromtxt(args.label_file, delimiter=',')

auc = roc_auc_score(labels, preds)

print('AUC: ', auc)
