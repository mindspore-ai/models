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
######################## eval lenet example ########################
Apply quantization-aware-training algorithms on LeNet5 model and eval accuracy according to model file:
"""

import os
from src.model_utils.config import config
from src.dataset import create_dataset
from src.lenet import LeNet5

import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from algorithm import create_simqat


def eval_lenet():
    print('eval with config: ', config)
    if config.device_target != "GPU":
        raise NotImplementedError("SimQAT only support running on GPU now!")
    if config.mode_name == "GRAPH":
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)

    network = LeNet5(config.num_classes)
    # apply golden stick algorithm on LeNet5 model
    algo = create_simqat()
    network = algo.apply(network)

    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    model = Model(network, net_loss, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset(os.path.join(config.data_path), config.batch_size)
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))


if __name__ == "__main__":
    eval_lenet()
