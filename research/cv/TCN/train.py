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
######################## train TCN net ########################
"""
import mindspore
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import set_seed
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.dataset import create_dataset
from src.datasetAP import create_datasetAP
from src.eval_call_back import EvalCallBack
from src.loss import NLLLoss
from src.lr_generator import get_lr
from src.metric import MyLoss
from src.model import TCN
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

set_seed(0)


def modelarts_pre_process():
    pass


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_net():
    """train net"""
    if config.dataset_name == 'permuted_mnist':
        train_dataset = create_dataset(config.train_data_path, config.batch_size)
        test_dataset = create_dataset(config.test_data_path, config.batch_size)
    elif config.dataset_name == 'adding_problem':
        train_dataset = create_datasetAP(config.train_data_path, config.N_train, config.seq_length, 'train',
                                         config.batch_train)
        test_dataset = create_datasetAP(config.test_data_path, config.N_test, config.seq_length, 'test',
                                        config.batch_test)

    net = TCN(config.channel_size, config.num_classes, [config.nhid] * config.level, config.kernel_size, config.dropout
              , config.dataset_name)
    lr = Tensor(get_lr(config, train_dataset.get_dataset_size()), dtype=mindspore.float32)
    net_opt = nn.Adam(net.trainable_params(), learning_rate=lr, weight_decay=config.weight_decay)

    if config.dataset_name == 'permuted_mnist':
        net_loss = NLLLoss(reduction='mean')
        model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    elif config.dataset_name == 'adding_problem':
        net_loss = nn.MSELoss()
        model = Model(net, net_loss, net_opt, metrics={"Accuracy": MyLoss()})

    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_tcn", directory=config.ckpt_path, config=config_ck)
    eval_per_epoch = 1
    epoch_per_eval = {"epoch": [], "acc": []}
    eval_cb = EvalCallBack(model, test_dataset, eval_per_epoch, epoch_per_eval)

    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=[time_cb, ckpoint_cb, LossMonitor(), eval_cb],
                dataset_sink_mode=config.dataset_sink_mode)


if __name__ == "__main__":
    train_net()
