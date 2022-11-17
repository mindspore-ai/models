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

import os
import datetime
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.common import set_seed

from src.dataset import classification_dataset
from src.c3d_model import C3D
from src.model_utils.config import config


class TestOneStepCell(nn.Cell):
    def __init__(self, network, criterion):
        super(TestOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.criterion = criterion

    def construct(self, data, label):
        output = self.network(data)
        loss = self.criterion(output, label)
        return output, loss

def test_net():
    '''run test'''
    config.load_type = 'test'
    set_seed(config.seed)
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=False,
                        device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=config.device_id)

    # logger
    config.outputs_dir = os.path.join(config.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    dataset, _ = classification_dataset(config.batch_size, 1, shuffle=True, repeat_num=1,
                                        drop_remainder=True)

    batch_num = dataset.get_dataset_size()

    # network
    print('start create network')
    network = C3D(config.num_classes)

    # pre_trained
    param_dict = load_checkpoint(config.ckpt_path)
    _ = load_param_into_net(network, param_dict)
    print('pre_trained model:', config.ckpt_path)

    network.set_train(mode=False)
    acc_sum, sample_num = 0, 0
    for index, (input_data, label) in enumerate(dataset):
        predictions = network(Tensor(input_data))
        predictions, label = predictions.asnumpy(), label.asnumpy()
        acc = np.sum(np.argmax(predictions, 1) == label[:, -1])
        batch_size = label.shape[0]
        acc_sum += acc
        sample_num += batch_size
        if index % 20 == 0:
            print("setep: {}/{}, acc: {}".format(index + 1, batch_num, acc / batch_size))

    accuracy_top1 = acc_sum / sample_num
    print('eval result: top_1 {:.3f}%'.format(accuracy_top1 * 100))


if __name__ == '__main__':
    test_net()
