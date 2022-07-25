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
######################## eval net ########################
"""

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.dataset import create_dataset_cifar10

import mindspore.nn as nn
from mindspore import context
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.serialization import load_checkpoint, load_param_into_net


def modelarts_process():
    config.ckpt_path = config.ckpt_file

def snn_model_build():
    """
    build snn model for lenet and resnet50
    """
    if config.net_name == "resnet50":
        from src.snn_resnet import snn_resnet50
        net = snn_resnet50(class_num=config.class_num)
    elif config.net_name == "lenet":
        from src.snn_lenet import snn_lenet
        net = snn_lenet(num_class=config.class_num)
    else:
        raise ValueError(f'config.model: {config.model_name} is not supported')
    return net


@moxing_wrapper(pre_process=modelarts_process)
def eval_net():
    """
    eval net
    """
    print('eval with config: ', config)
    if config.mode_name == 'GRAPH':
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    context.set_context(device_id=config.device_id)
    ds_eval = create_dataset_cifar10(data_path=config.data_path, do_train=False, batch_size=config.batch_size)
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")
    network_eval = snn_model_build()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    model = Model(network_eval, net_loss, metrics={"Accuracy": Accuracy()})
    param_dict = load_checkpoint(config.ckpt_path)
    load_param_into_net(network_eval, param_dict)
    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))

if __name__ == "__main__":
    eval_net()
