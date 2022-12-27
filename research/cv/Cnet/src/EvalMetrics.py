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
import numpy as np
import mindspore.ops as ops
from mindspore.nn.metrics.metric import Metric, rearrange_inputs


class ErrorRateAt95Recall(Metric):

    def __init__(self):
        super(ErrorRateAt95Recall, self).__init__()
        self.clear()

    def clear(self):
        self.distances = []
        self.labels = []
        self.num_tests = 0

    @rearrange_inputs
    def update(self, *inputs):
        distance = self._convert_data(inputs[0])
        ll = self._convert_data(inputs[1])
        self.distances.append(distance)
        self.labels.append(ll)
        self.num_tests += ll.shape[0]

    def eval(self):
        distances = np.vstack(self.distances).reshape(self.num_tests)
        labels = np.vstack(self.labels).reshape(self.num_tests)
        recall_point = 0.95
        labels = labels[np.argsort(distances)]
        threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

        FP = np.sum(labels[:threshold_index] == 0)  # Below threshold (i.e., labelled positive), but should be negative
        TN = np.sum(labels[threshold_index:] == 0)  # Above threshold (i.e., labelled negative), and should be negative
        return float(FP) / float(FP + TN)


def inference(network, eval_data, acc_fn):
    sqrt = ops.Sqrt()
    reduce_sum = ops.ReduceSum()
    reshape = ops.Reshape()
    for data in eval_data.create_dict_iterator():
        out_a = network(data['data_a'])
        out_p = network(data['data_p'])
        dists = sqrt(reduce_sum((out_a - out_p) ** 2, 1))  # euclidean distance
        acc_fn.update(reshape(dists, (-1, 1)), reshape(data['label'], (-1, 1)))
    acc = acc_fn.eval()
    acc_fn.clear()
    return acc
