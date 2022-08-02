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
"""TB-Net training."""

import os
import math

import numpy as np
from mindspore import context, Model, Tensor
from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback import Callback, TimeMonitor
import mindspore.common.dtype as mstype

from src import tbnet, config, metrics, dataset

from src.utils.param import param
from src.utils.moxing_adapter import moxing_wrapper
from preprocess_dataset import preprocess_data


class MyLossMonitor(Callback):
    """My loss monitor definition."""

    def on_train_epoch_end(self, run_context):
        """Print loss at each epoch end."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        print('loss:' + str(loss))

    def on_eval_epoch_end(self, run_context):
        """Print loss at each epoch end."""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        print('loss:' + str(loss))


@moxing_wrapper(preprocess_data)
def train_tbnet():
    """Training process."""
    config_path = os.path.join(param.data_path, 'data', param.dataset, 'config.json')
    train_csv_path = os.path.join(param.data_path, 'data', param.dataset, param.train_csv)
    test_csv_path = os.path.join(param.data_path, 'data', param.dataset, param.test_csv)
    ckpt_path = param.load_path

    context.set_context(device_id=param.device_id)
    if param.run_mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=param.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=param.device_target)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    print(f"creating dataset from {train_csv_path}...")
    net_config = config.TBNetConfig(config_path)
    if param.device_target == 'Ascend':
        net_config.per_item_paths = math.ceil(net_config.per_item_paths / 16) * 16
        net_config.embedding_dim = math.ceil(net_config.embedding_dim / 16) * 16
    train_ds = dataset.create(train_csv_path, net_config.per_item_paths, train=True).batch(net_config.batch_size)
    test_ds = dataset.create(test_csv_path, net_config.per_item_paths, train=True).batch(net_config.batch_size)

    print("creating TBNet for training...")
    network = tbnet.TBNet(net_config)
    loss_net = tbnet.NetWithLossClass(network, net_config)
    if param.device_target == 'Ascend':
        loss_net.to_float(mstype.float16)
        train_net = tbnet.TrainStepWrap(loss_net, net_config.lr, loss_scale=True)
    else:
        train_net = tbnet.TrainStepWrap(loss_net, net_config.lr)

    train_net.set_train()
    eval_net = tbnet.PredictWithSigmoid(network)
    time_callback = TimeMonitor(data_size=train_ds.get_dataset_size())
    loss_callback = MyLossMonitor()
    model = Model(network=train_net, eval_network=eval_net, metrics={'auc': metrics.AUC(), 'acc': metrics.ACC()})
    print("training...")
    for i in range(param.epochs):
        print(f'===================== Epoch {i} =====================')
        model.train(epoch=1, train_dataset=train_ds, callbacks=[time_callback, loss_callback], dataset_sink_mode=False)
        train_out = model.eval(train_ds, dataset_sink_mode=False)
        test_out = model.eval(test_ds, dataset_sink_mode=False)
        print(f'Train AUC:{train_out["auc"]} ACC:{train_out["acc"]}  Test AUC:{test_out["auc"]} ACC:{test_out["acc"]}')

        if i >= param.epochs - 5:
            if param.enable_modelarts:
                save_checkpoint(network, os.path.join(param.output_path, f'tbnet_epoch{i}.ckpt'))
            else:
                save_checkpoint(network, os.path.join(ckpt_path, f'tbnet_epoch{i}.ckpt'))


if __name__ == '__main__':
    train_tbnet()
