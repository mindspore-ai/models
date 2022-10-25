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
""" eval_callback """
from mindspore.nn.metrics.metric import Metric
from src.utils.aux_loss import SoftmaxCrossEntropyLoss


class pspnet_metric(Metric):
    """ callback class """
    def __init__(self, num_classes=150, ignore_label=255):
        super(pspnet_metric, self).__init__()
        self.loss = SoftmaxCrossEntropyLoss(num_classes, ignore_label)
        self.val_loss = 0
        self.count = 0
        self.clear()

    def clear(self):
        """ clear the init value """
        self.val_loss = 0
        self.count = 0

    def update(self, *inputs):
        """ update the calculate process """
        if len(inputs) != 2:
            raise ValueError('Expect inputs (y_pred, y), but got {}'.format(len(inputs)))
        _, predict = inputs[0]
        the_loss = self.loss(predict, inputs[1])
        self.val_loss += the_loss
        self.count += 1

    def eval(self):
        """ return the result """
        return self.val_loss / float(self.count)
