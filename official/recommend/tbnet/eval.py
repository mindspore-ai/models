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
"""TB-Net evaluation."""

import os
import math

from mindspore import context, Model, load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype

from src import tbnet, config, metrics, dataset

from src.utils.param import param
from src.utils.moxing_adapter import moxing_wrapper
from preprocess_dataset import preprocess_data


@moxing_wrapper(preprocess_data)
def eval_tbnet():
    """Evaluation process."""
    config_path = os.path.join(param.data_path, 'data', param.dataset, 'config.json')
    test_csv_path = os.path.join(param.data_path, 'data', param.dataset, param.test_csv)
    ckpt_path = param.load_path

    context.set_context(device_id=param.device_id)
    if param.run_mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=param.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=param.device_target)

    print(f"creating dataset from {test_csv_path}...")
    net_config = config.TBNetConfig(config_path)
    if param.device_target == 'Ascend':
        net_config.per_item_paths = math.ceil(net_config.per_item_paths / 16) * 16
        net_config.embedding_dim = math.ceil(net_config.embedding_dim / 16) * 16
    eval_ds = dataset.create(test_csv_path, net_config.per_item_paths, train=True).batch(net_config.batch_size)

    print(f"creating TBNet from checkpoint {param.checkpoint_id} for evaluation...")
    network = tbnet.TBNet(net_config)
    if param.device_target == 'Ascend':
        network.to_float(mstype.float16)
    param_dict = load_checkpoint(os.path.join(ckpt_path, f'tbnet_epoch{param.checkpoint_id}.ckpt'))
    load_param_into_net(network, param_dict)

    loss_net = tbnet.NetWithLossClass(network, net_config)
    train_net = tbnet.TrainStepWrap(loss_net, net_config.lr)
    train_net.set_train()
    eval_net = tbnet.PredictWithSigmoid(network)
    model = Model(network=train_net, eval_network=eval_net, metrics={'auc': metrics.AUC(), 'acc': metrics.ACC()})

    print("evaluating...")
    e_out = model.eval(eval_ds, dataset_sink_mode=False)
    print(f'Test AUC:{e_out["auc"]} ACC:{e_out["acc"]}')
    if param.enable_modelarts:
        with open(os.path.join(param.output_path, 'result.txt'), 'w') as f:
            f.write(f'Test AUC:{e_out["auc"]} ACC:{e_out["acc"]}')


if __name__ == '__main__':
    eval_tbnet()
