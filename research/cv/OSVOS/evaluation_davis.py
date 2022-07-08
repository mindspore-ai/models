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
"""DAVIS2016 eval."""
import os
import argparse
import numpy as np
from src.utils import Evaluation


parser = argparse.ArgumentParser(description='DAVIS2016 eval running')
parser.add_argument("--eval_txt", type=str, default=None, help="DAVIS2016 evaluation data list")
parser.add_argument("--prediction_path", type=str, default=None, help="the prediction images path")
parser.add_argument("--gt_path", type=str, default=None, help="the gt masks path")

def main():
    args = parser.parse_args()
    mask_path = os.path.join(args.gt_path, 'Annotations/480p/')

    metric_eval = Evaluation(args.eval_txt, args.prediction_path, mask_path)
    metrics_J, metrics_F = metric_eval.evaluate()
    print('metrics_J:', metrics_J)
    print('metrics_F:', metrics_F)
    list_J = []
    list_F = []
    for key, value in metrics_J.items():
        list_J.extend(value)
        list_F.extend(metrics_F[key])

    avg_J = np.sum(np.array(list_J)/len(list_J))
    avg_F = np.sum(np.array(list_F)/len(list_F))

    print('jaccard:', avg_J)
    print('f1:', avg_F)
    print('J&F:', (avg_J + avg_F)/2)

if __name__ == '__main__':
    main()
