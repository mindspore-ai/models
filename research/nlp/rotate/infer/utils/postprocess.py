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
postprocess
"""

import sys
import numpy as np
from tqdm import tqdm

result_path = sys.argv[1]

argsort = np.loadtxt(result_path+'/argsort.txt', dtype=int)
positive_arg = np.loadtxt(result_path+'/positive_arg.txt', dtype=int)
logs = []

for i in tqdm(range(argsort.shape[0])):
    ranking = np.where(argsort[i, :] == positive_arg[i])[0][0]
    ranking = 1 + ranking
    logs.append({
        'MRR': 1.0 / ranking,
        'MR': ranking,
        'HITS@1': 1.0 if ranking <= 1 else 0.0,
        'HITS@3': 1.0 if ranking <= 3 else 0.0,
        'HITS@10': 1.0 if ranking <= 10 else 0.0,
    })

metrics = {}
for metric in logs[0].keys():
    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

print(metrics)
