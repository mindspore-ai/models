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
        if not self.labels:
            raise RuntimeError('labels must not be 0.')
        distances = np.vstack(self.distances).reshape(self.num_tests)
        labels = np.vstack(self.labels).reshape(self.num_tests)
        recall_point = 0.95
        labels = labels[np.argsort(distances)]
        threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))
        FP = np.sum(labels[:threshold_index] == 0)  # Below threshold (i.e., labelled positive), but should be negative
        TN = np.sum(labels[threshold_index:] == 0)  # Above threshold (i.e., labelled negative), and should be negative
        return float(FP) / float(FP + TN)


class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
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
        if not self.labels:
            raise RuntimeError('labels must not be 0.')
        distances = np.vstack(self.distances).reshape(self.num_tests)
        labels = np.vstack(self.labels).reshape(self.num_tests)
        pred_labels = labels[np.argsort(distances)]
        array = np.zeros(len(labels))
        _sum = np.sum(labels)
        array[:_sum] = np.ones(_sum)
        acc = np.sum(np.abs(np.array(array - pred_labels))) / len(labels)
        acc = 1 - acc
        return acc


def inference(network, eval_data, fpr95, acc_fn):
    op_sqrt = ops.Sqrt()
    op_sum = ops.ReduceSum()
    op_reshape = ops.Reshape()
    for data in eval_data.create_dict_iterator():
        opt_e, sar_e, _, _, _, _ = network(data['opt_img'], data['sar_img'])
        dists = op_sqrt(op_sum((opt_e - sar_e) ** 2, 1))
        fpr95.update(op_reshape(dists, (-1, 1)), op_reshape(data['label'], (-1, 1)))
        acc_fn.update(op_reshape(dists, (-1, 1)), op_reshape(data['label'], (-1, 1)))
    acc_fpr95 = fpr95.eval()
    acc = acc_fn.eval()
    fpr95.clear()
    acc_fn.clear()
    return acc_fpr95, acc
