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
"""postprocess_infer data"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

parser = argparse.ArgumentParser(description='Postprocess of Fasttext Inference')
parser.add_argument('--target_label_path', type=str)
parser.add_argument('--predict_label_path', type=str)
args = parser.parse_args()

target_sens = np.loadtxt(args.target_label_path, dtype=np.int32).reshape(-1, 1)
predictions = np.loadtxt(args.predict_label_path, dtype=np.int32).reshape(-1, 1)

target_sens = np.array(target_sens).flatten()
merge_target_sens = []
target_label1 = ['0', '1', '2', '3']
for target_sen in target_sens:
    merge_target_sens.extend([target_sen])
target_sens = merge_target_sens
predictions = np.array(predictions).flatten()
merge_predictions = []
for prediction in predictions:
    merge_predictions.extend([prediction])
predictions = merge_predictions
acc = accuracy_score(target_sens, predictions)

result_report = classification_report(target_sens, predictions, target_names=target_label1)
print("********Accuracy: ", acc)
print(result_report)
