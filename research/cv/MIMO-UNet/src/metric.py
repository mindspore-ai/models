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
Metrics
"""

from mindspore import nn
from mindspore import ops
from skimage.metrics import peak_signal_noise_ratio


class PSNR(nn.Metric):
    """peak signal-noise ratio"""
    def __init__(self):
        super().__init__()
        self.ops_sqrt = ops.Sqrt()
        self.ops_max = ops.ReduceMax()
        self.ops_log = ops.Log()
        self.ops_se = ops.SquaredDifference()
        self.ops_mean = ops.ReduceMean()
        self.eps = 0.
        self.psnr = 0.
        self.total_num = 0

    def clear(self):
        """clear"""
        self.psnr = 0.
        self.total_num = 0

    def eval(self):
        """eval"""
        if self.total_num == 0:
            return 0
        return self.psnr / self.total_num

    def update(self, pred, label):
        """update"""
        pred_numpy = pred[2].asnumpy()
        label_numpy = label.asnumpy()
        self.psnr += peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
        self.total_num += 1
