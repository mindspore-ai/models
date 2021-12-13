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
"""post process for 310 inference."""
import os
import shutil
import re
import numpy as np
from sklearn.metrics import roc_auc_score

from src.model_utils.config import config

from mindspore import Tensor
from mindspore.nn.metrics import Metric


class AUCMetric(Metric):
    """Metric method"""

    def __init__(self):
        super(AUCMetric, self).__init__()
        self.pred_probs = []
        self.true_labels = []

    def clear(self):
        """Clear the internal evaluation result."""
        self.pred_probs = []
        self.true_labels = []

    def update(self, *inputs):
        batch_predict = inputs[0].asnumpy()
        batch_label = inputs[1].asnumpy()
        self.pred_probs.extend(batch_predict.flatten().tolist())
        self.true_labels.extend(batch_label.flatten().tolist())

    def eval(self):
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError('true_labels.size() is not equal to pred_probs.size()')
        auc = roc_auc_score(self.true_labels, self.pred_probs)
        return auc


def ger_path(path):
    """generate income path and marital path"""
    return path + '/income_path', path + '/marital_path'


def mdir(path1, path2):
    """makedir"""
    f = os.path.exists(path1)
    if not f:
        os.makedirs(path1)

    f = os.path.exists(path2)
    if not f:
        os.makedirs(path2)


def mycopy(dirlist, path, income_path, marital_path):
    """copy files"""
    for i in dirlist[:]:
        x = re.search('census_val__.*_0.bin', i)
        if x:
            v, _ = x.group().split('.')
            v = v[:-2] + '.bin'
            # print(v)
            shutil.copyfile(path + x.group(), income_path + v)
        x = re.search('census_val__.*_1.bin', i)
        if x:
            v, _ = x.group().split('.')
            v = v[:-2] + '.bin'
            shutil.copyfile(path + x.group(), marital_path + v)


def get_acc():
    """get accuracy"""
    path = config.result_bin_path
    print('-------------' + path)
    dirlist = os.listdir(path)

    income_path, marital_path = ger_path(path)
    mdir(income_path, marital_path)

    mycopy(dirlist, path + '/', income_path + '/', marital_path + '/')
    print('copy success')


def auc_com(rst_path, original_path):
    """auc class call function"""
    labellist = os.listdir(original_path)
    print(rst_path, '<------>', original_path)
    auc_metric = AUCMetric()
    for v in labellist:
        r = rst_path + '/' + v
        l = original_path + '/' + v
        logit = Tensor(np.fromfile(r, np.float16).reshape(2, 1))
        label = Tensor(np.fromfile(l, np.float16).reshape(2, 1))
        res = [logit, label]
        auc_metric.update(*res)
    auc = auc_metric.eval()
    print('auc:  ', auc)


if __name__ == "__main__":
    get_acc()
    auc_com(config.result_bin_path + '/income_path', config.label1_path)
    auc_com(config.result_bin_path + '/marital_path', config.label2_path)
