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
"""export."""

import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.nets import net_factory

from model_utils.config import config
from model_utils.device_adapter import get_device_id

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False,
                    device_id=get_device_id())


class BuildEvalNetwork(nn.Cell):
    """BuildEvalNetwork"""
    def __init__(self, network, input_format="NCHW"):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)
        self.transpose = ops.Transpose()
        self.format = input_format

    def construct(self, input_data):
        if self.format == "NHWC":
            input_data = self.transpose(input_data, (0, 3, 1, 2))
        output = self.network(input_data)
        output = self.softmax(output)
        return output

def run_export():
    """export"""
    args = config

    # network
    if args.model == 'deeplab_v3_s16':
        network = net_factory.nets_map[args.model]('eval', args.num_classes, 16, args.freeze_bn)
    elif args.model == 'deeplab_v3_s8':
        network = net_factory.nets_map[args.model]('eval', args.num_classes, 8, args.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(args.model))

    eval_net = BuildEvalNetwork(network, args.input_format)

    # load model
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)

    if config.input_format == "NHWC":
        input_data = Tensor(
            np.ones([config.export_batch_size, config.input_size, config.input_size, 3]).astype(np.float32))
    else:
        input_data = Tensor(
            np.ones([config.export_batch_size, 3, config.input_size, config.input_size]).astype(np.float32))
    export(network, input_data, file_name=config.file_name, file_format=config.file_format)



if __name__ == '__main__':
    run_export()
