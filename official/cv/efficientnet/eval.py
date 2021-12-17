# Copyright 2020 Huawei Technologies Co., Ltd
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
"""evaluate imagenet"""
import time

from mindspore import nn
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config
from src.dataset import create_dataset_val
from src.efficientnet import efficientnet_b0, efficientnet_b1
from src.loss import LabelSmoothingCrossEntropy

if __name__ == '__main__':
    if config.model == 'efficientnet_b0':
        model_name = 'efficientnet_b0'
    elif config.model == 'efficientnet_b1':
        model_name = 'efficientnet_b1'
    else:
        raise NotImplementedError("This model currently not supported")

    context.set_context(mode=context.GRAPH_MODE, device_target=config.platform)

    if model_name == 'efficientnet_b0':
        net = efficientnet_b0(num_classes=config.num_classes,
                              cfg=config,
                              drop_rate=config.drop,
                              drop_connect_rate=config.drop_connect,
                              global_pool=config.gp,
                              bn_tf=config.bn_tf,
                             )
    elif model_name == 'efficientnet_b1':
        net = efficientnet_b1(num_classes=config.num_classes,
                              cfg=config,
                              drop_rate=config.drop,
                              drop_connect_rate=config.drop_connect,
                              global_pool=config.gp,
                              bn_tf=config.bn_tf,
                             )

    ckpt = load_checkpoint(config.checkpoint)
    load_param_into_net(net, ckpt)
    net.set_train(False)

    if config.dataset == 'ImageNet':
        data_url = config.data_path
        train_data_url = data_url + '/train'
        val_data_url = data_url + '/val'
    elif config.dataset == 'CIFAR10':
        data_url = config.data_path
        train_data_url = data_url
        val_data_url = data_url

    val_dataset = create_dataset_val(
        config.dataset, model_name, val_data_url, config.batch_size,
        workers=config.workers, distributed=False)
    loss = LabelSmoothingCrossEntropy(smooth_factor=config.smoothing, num_classes=config.num_classes)
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss, optimizer=None, metrics=eval_metrics)

    dataset_sink_mode = config.platform != "CPU"

    start_time = time.time()
    metrics = model.eval(val_dataset, dataset_sink_mode=dataset_sink_mode)
    end_time = time.time()

    time_taken = end_time - start_time

    print("\nEvaluation results:")
    print("Metrics:", metrics)
    print(f"Time taken: {int(time_taken*1000)} ms")
