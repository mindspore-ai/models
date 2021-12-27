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
"""postprocess."""
import os
import numpy as np
from mindspore import Tensor
from src.dlrm import AccMetric
from src.model_utils.config import config


def get_acc():
    ''' get accuracy '''
    acc_metric = AccMetric()
    files = os.listdir(config.label_path) # preprocessed files path
    batch_size = config.test_batch_size

    for f in files:
        rst_file = os.path.join(config.result_path, f.split('.')[0] + '_0.bin')  # infer result file path
        label_file = os.path.join(config.label_path, f)

        logit = Tensor(np.fromfile(rst_file, np.float32).reshape(batch_size, 1))
        label = Tensor(np.fromfile(label_file, np.float32).reshape(batch_size, 1))

        res = []
        res.append(logit)
        res.append(label)

        acc_metric.update(*res)
    acc = acc_metric.eval()
    print("auc : {}".format(acc))

if __name__ == '__main__':
    get_acc()
