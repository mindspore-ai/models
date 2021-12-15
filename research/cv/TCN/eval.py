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
"""
######################## eval TCN example ########################
eval TCN according to model file:
python eval.py --test_data_path /YourDataPath --ckpt_file Your.ckpt
"""
import mindspore
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_dataset
from src.datasetAP import create_datasetAP
from src.loss import NLLLoss
from src.lr_generator import get_lr
from src.metric import MyLoss
from src.model import TCN
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


@moxing_wrapper()
def eval_net():
    """eval function"""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    net = TCN(config.channel_size, config.num_classes, [config.nhid] * config.level, config.kernel_size, config.dropout
              , config.dataset_name)
    if config.dataset_name == 'permuted_mnist':
        test_dataset = create_dataset(config.test_data_path, config.batch_size)
    elif config.dataset_name == 'adding_problem':
        test_dataset = create_datasetAP(config.test_data_path, config.N_test, config.seq_length, 'test',
                                        config.batch_test)
    lr = Tensor(get_lr(config, test_dataset.get_dataset_size()), dtype=mindspore.float32)

    net_opt = nn.Adam(net.trainable_params(), learning_rate=lr, weight_decay=config.weight_decay)
    if config.dataset_name == 'permuted_mnist':
        net_loss = NLLLoss(reduction='mean')
        model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    elif config.dataset_name == 'adding_problem':
        net_loss = nn.MSELoss()
        model = Model(net, net_loss, net_opt, metrics={"Accuracy": MyLoss()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)

    acc = model.eval(test_dataset)
    print("============== {} ==============".format(acc))


if __name__ == "__main__":
    eval_net()
