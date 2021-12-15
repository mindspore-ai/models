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

"""CTC Loss."""
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
import numpy as np


class CTC_Loss(nn.Cell):
    """
         CTCLoss definition
         Args:
            batch_soze(int): batch_size
            max_label_length(int): max number of label length for each input.
    """

    def __init__(self, batch_size, max_label_length):
        super(CTC_Loss, self).__init__()
        labels_indices = []
        for i in range(batch_size):
            for j in range(max_label_length):
                labels_indices.append([i, j])
        self.labels_indices = Parameter(Tensor(np.array(labels_indices), mstype.int64), name="labels_indices")
        self.reshape = P.Reshape()
        self.ctc_loss = P.CTCLoss(preprocess_collapse_repeated=False,
                                  ctc_merge_repeated=True,
                                  ignore_longer_outputs_than_inputs=True)
        self.reduce_mean = P.ReduceMean()

    def construct(self, logit, label, seq_len):
        label_values = self.reshape(label, (-1,))
        loss, _ = self.ctc_loss(logit, self.labels_indices, label_values, seq_len)
        return self.reduce_mean(loss)
