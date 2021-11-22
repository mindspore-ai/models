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
import time
import numpy as np
import scipy.stats

import mindspore.nn as nn
import mindspore.ops as P
from mindspore.train.callback import Callback

class EmdLoss(nn.Cell):
    def __init__(self):
        super(EmdLoss, self).__init__()
        self.square = P.Square()
        self.reduce_mean = P.ReduceMean()
        self.cumsum = P.CumSum()
        self.sqrt = P.Sqrt()
    def construct(self, data, label):
        data = self.cumsum(data, 1)
        label = self.cumsum(label, 1)
        diff = data - label
        emd = self.sqrt(self.reduce_mean(self.square(diff), 1))
        return self.reduce_mean(emd)

class PrintFps(Callback):
    def __init__(self, train_data_num, start_time, end_time):
        self.train_data_num = train_data_num
        self.start_time = start_time
        self.end_time = end_time
    def epoch_begin(self, run_context):
        self.start_time = time.time()
    def epoch_end(self, run_context):
        self.end_time = time.time()
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        fps = self.train_data_num / (self.end_time - self.start_time)
        loss = cb_param.net_outputs
        used_time = self.end_time - self.start_time
        print("Epoch:{} ,used time is:{:.2f}, fps: {:.2f}imgs/sec".format(cur_epoch, used_time, fps))
        print('Step_end loss is', loss)

class spearman(nn.Accuracy):

    def clear(self):
        self._correct_num = []
        self._total_num = []

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        SCORE_LIST = [[i for i in range(1, 11)]] * inputs[0].shape[0]
        gt = np.sum(y * np.array(SCORE_LIST), axis=1)
        score = np.sum(y_pred * np.array(SCORE_LIST), axis=1)
        self._correct_num += gt.tolist()
        self._total_num += score.tolist()

    def eval(self):
        return scipy.stats.spearmanr(self._correct_num, self._total_num)
