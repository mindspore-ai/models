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
"""Train NIMA vgg16"""

import os
import time

import mindspore as ms
import mindspore.communication.management as D
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Model
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor

from src.MyCallBack import EvalCallBack
from src.MyDataset import create_dataset
from src.MyMetric import EmdLoss
from src.MyMetric import PrintFps
from src.MyMetric import Spearman
from src.config import config
from src.vgg import vgg16


def train_net(model, args, ds_train_, steps_per_epoch_train_):
    """Train network"""
    ds_val, _ = create_dataset(args, data_mode='val')
    print('steps_per_epoch_train', steps_per_epoch_train_, 'epoch_size', args.epoch_size)
    eval_per_epoch = 1
    call_backs = []
    print("============== Starting Training ==============")
    if args.device_num == 1:
        epoch_per_eval = {"epoch": [], "spearman": []}
        eval_cb = EvalCallBack(model, ds_val, eval_per_epoch, epoch_per_eval)
        call_backs.append(eval_cb)

    if args.device_num == 1 or args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch_train_,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=args.ckpt_filename, directory=args.ckpt_save_dir, config=config_ck)
        call_backs.append(ckpoint_cb)

    train_data_num = steps_per_epoch_train_ * args.batch_size
    init_time = time.time()
    fps = PrintFps(train_data_num, init_time, init_time)
    time_cb = TimeMonitor(train_data_num)
    loss_cb = LossMonitor()
    call_backs.extend([fps, time_cb, loss_cb])
    model.train(args.epoch_size, ds_train_, callbacks=call_backs, dataset_sink_mode=False)


def load_vgg_weights(network, ckpt_weights):
    """Load vgg weight in model"""
    classifier_weights = ckpt_weights.pop('classifier.6.weight').asnumpy()[:10]
    classifier_bias = ckpt_weights.pop('classifier.6.bias').asnumpy()[:10]
    ckpt_weights['classifier.6.weight'] = ms.Parameter(classifier_weights)
    ckpt_weights['classifier.6.bias'] = ms.Parameter(classifier_bias)
    ckpt_weights.pop('momentum')
    ckpt_weights.pop('global_step')
    ckpt_weights.pop('learning_rate')
    non_model_keys = [
        key_name for key_name in ckpt_weights.keys() if 'moments' in key_name
    ]
    for key in non_model_keys:
        ckpt_weights.pop(key)
    ms.load_param_into_net(network, ckpt_weights, strict_load=False)


if __name__ == "__main__":
    if not os.path.exists(config.ckpt_save_dir):
        os.mkdir(config.ckpt_save_dir)
    if config.enable_modelarts:
        import moxing as mox
        mox.file.shift('os', 'mox')
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.distribute == 'true':
        D.init('nccl')
        device_num = D.get_group_size()
        rank = D.get_rank()
        config.device_num = device_num
        config.rank = rank
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        device_id = config.device_id
        config.device_num = 1
        config.rank = device_id
        context.set_context(device_id=device_id)

    print('batch_size:', config.batch_size, 'workers:', config.num_parallel_workers)
    print('device_id', config.rank, 'device_num', config.device_num)
    set_seed(10)

    net = vgg16(num_classes=10, args=config)
    ckpt = ms.load_checkpoint(config.checkpoint_path)
    print('loading weights from ', config.checkpoint_path)
    load_vgg_weights(net, ckpt)

    #dataset
    ds_train, steps_per_epoch_train = create_dataset(config, data_mode='train')

    # loss
    criterion = EmdLoss()
    # opt
    learning_rate = config.learning_rate
    learning_rate = nn.exponential_decay_lr(
        learning_rate=learning_rate,
        decay_rate=0.95,
        total_step=steps_per_epoch_train * config.epoch_size,
        decay_epoch=10,
        step_per_epoch=steps_per_epoch_train,
    )
    momentum = config.momentum
    weight_decay = config.weight_decay
    opt = nn.SGD(
        [
            {
                'params': [param for param in net.trainable_params() if 'classifier.6' in param.name],
                'lr': ms.Tensor(learning_rate, dtype=ms.float32) * 10,
            },
            {
                'params': [param for param in net.trainable_params() if 'classifier.6' not in param.name],
                'lr': ms.Tensor(learning_rate, dtype=ms.float32),
            }
        ],
        weight_decay=weight_decay,
        momentum=momentum,
    )
    # Construct model
    metrics = {'spearman': Spearman()}
    net = Model(net, criterion, opt, metrics=metrics)
    # Train
    train_net(net, config, ds_train, steps_per_epoch_train)
    if config.enable_modelarts:
        for file in os.listdir(config.ckpt_save_dir):
            mox.file.copy(os.path.join(config.ckpt_save_dir, file),
                          os.path.join(config.output_path, 'Ascend_{}P_'.format(device_num) + file))
