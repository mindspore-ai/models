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
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
import numpy as np
class GumbleSoftmax(nn.Cell):
    def __init__(self, hard=True):
        super(GumbleSoftmax, self).__init__()
        self.training = False
        self.hard = hard
        self.gpu = False
        self.print = P.Print()
        self.minval = mindspore.Tensor(0, mindspore.float32)
        self.maxval = mindspore.Tensor(1, mindspore.float32)
        self.updates = mindspore.Tensor(np.array([1.0]), mindspore.float32)
    def cuda(self):
        self.gpu = True
    def cpu(self):
        self.gpu = False
    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = mindspore.ops.uniform(template_tensor.shape, self.minval,
                                                       self.maxval, dtype=mindspore.float32)
        uniform_samples_tensor = uniform_samples_tensor.abs()
        gumble_samples_tensor = - mindspore.ops.Log()(eps - mindspore.ops.Log()(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gumble_samples_tensor = self.sample_gumbel_like(logits)
        gumble_trick_log_prob_samples = logits + gumble_samples_tensor
        soft_samples = mindspore.ops.Softmax(-1)(gumble_trick_log_prob_samples)
        return soft_samples
    def gumbel_softmax(self, logits, hard=True):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits)
        y_hard = []
        max_value_indexes = []
        if hard:
            zeroslike = mindspore.ops.ZerosLike()
            max_value_indexes = y.argmax(axis=1)
            y_hard = zeroslike(logits)
            for batch_idx in range(logits.shape[0]):
                y_hard[batch_idx][max_value_indexes[batch_idx]] = 1.0
        return y_hard
    def construct(self, logits, temp=1, force_hard=True):
        result = 0
        if self.training and not force_hard:
            result = self.gumbel_softmax(logits, hard=False)
        else:
            result = self.gumbel_softmax(logits, hard=True)
        return result
