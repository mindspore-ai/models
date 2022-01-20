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
'''post process for 310 inference'''
import argparse
import numpy as np
import scipy.stats
import scipy.special as sc
parser = argparse.ArgumentParser(description='postprocess for relationnet')
parser.add_argument("--result_path", type=str, default="ascend310_infer/out/result_Files", help="result file path")
parser.add_argument("--label_path", type=str, default="./data/label", help="label path")
args = parser.parse_args()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sc.stdtrit(n-1, (1+confidence)/2.)
    return m, h

accuracies = []
class_num = 5
sample_num_per_class = 1
for i in range(0, 1000):
    total_rewards = 0
    test_relations = np.fromfile(args.result_path+"/a"+str(i)+".bin", dtype=np.float32)
    predict_labels = np.argmax(test_relations.reshape(5, 5), axis=1)
    test_batch_labels = np.fromfile(args.label_path+"/b"+str(i)+".bin", dtype=np.int32)
    rewards = [1 if predict_labels[j] == test_batch_labels[j] else 0 for j in range(class_num)]
    total_rewards += np.sum(rewards)
    accuracy = np.sum(rewards) / 1.0 / class_num / sample_num_per_class
    accuracies.append(accuracy)
accuracies, _ = mean_confidence_interval(accuracies, confidence=0.95)
print("aver_accuracy : %.4f"%(accuracies))
