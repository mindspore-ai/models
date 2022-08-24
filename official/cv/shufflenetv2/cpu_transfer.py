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
"""train ShuffleNetV2"""
import argparse
import ast
import time
from mindspore import context
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore import nn
from src.lr_generator import get_lr_basic
from src.shufflenetv2 import ShuffleNetV2
from src.dataset import create_dataset
from src.CrossEntropySmooth import CrossEntropySmooth
from src.config import config_cpu

set_seed(1)

def modelarts_pre_process():
    pass

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image classification transformation')
    parser.add_argument('--checkpoint_input_path', type=str, default='',
                        help='the checkpoint of ShuffleNetV2 (Default: None)')
    parser.add_argument('--checkpoint_save_path', type=str, default='',
                        help='the directory that saves the ckpt')
    parser.add_argument('--train_dataset', type=str, default='',
                        help='the training data for transformation')
    parser.add_argument('--use_pynative_mode', type=ast.literal_eval, default=False,
                        help='whether to use pynative mode for device(Default: False)')
    parser.add_argument('--platform', type=str, default='CPU', choices=('Ascend', 'GPU', 'CPU'),
                        help='run platform(Default:Ascend)')
    args_opt = parser.parse_args()
    if args_opt.use_pynative_mode:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.platform,
                            device_id=config_cpu.device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform,
                            device_id=config_cpu.device_id, save_graphs=False)

    # define network
    net = ShuffleNetV2(model_size=config_cpu.model_size, n_class=config_cpu.num_classes)

    # define dataset
    train_dataset = create_dataset(args_opt.train_dataset, True, config_cpu.rank, config_cpu.group_size,
                                   num_parallel_workers=config_cpu.num_parallel_workers,
                                   batch_size=config_cpu.train_batch_size,
                                   drop_remainder=config_cpu.drop_remainder, shuffle=True,
                                   cutout=config_cpu.cutout, cutout_length=config_cpu.cutout_length,
                                   normalize=config_cpu.normalize,
                                   enable_tobgr=config_cpu.enable_tobgr)

    # load ckpt
    if args_opt.checkpoint_input_path:
        ckpt = load_checkpoint(args_opt.checkpoint_input_path)
        if config_cpu.remove_classifier_parameter:
            filter_list = [x.name for x in net.classifier.get_parameters()]
            filter_checkpoint_parameter_by_list(ckpt, filter_list)
        load_param_into_net(net, ckpt)

    # loss
    if not config_cpu.use_nn_default_loss:
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config_cpu.label_smooth_factor,
                                  num_classes=config_cpu.num_classes)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # get learning rate
    batches_per_epoch = train_dataset.get_dataset_size()
    lr = get_lr_basic(lr_init=config_cpu.lr_init, total_epochs=config_cpu.epoch_size,
                      steps_per_epoch=batches_per_epoch, is_stair=True)
    lr = Tensor(lr)

    # define optimization
    optimizer = Momentum(params=net.trainable_params(),
                         learning_rate=lr,
                         momentum=config_cpu.momentum,
                         weight_decay=config_cpu.weight_decay)

    # model
    loss_scale_manager = FixedLossScaleManager(config_cpu.loss_scale, drop_overflow_update=False)
    model = Model(net,
                  loss_fn=loss,
                  optimizer=optimizer,
                  amp_level=config_cpu.amp_level,
                  metrics={'acc'},
                  loss_scale_manager=loss_scale_manager)

    # define callbacks
    cb = [TimeMonitor(), LossMonitor()]
    if args_opt.checkpoint_save_path:
        save_ckpt_path = args_opt.checkpoint_save_path
        config_ck = CheckpointConfig(save_checkpoint_steps=config_cpu.save_checkpoint_epochs * batches_per_epoch,
                                     keep_checkpoint_max=config_cpu.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint("shufflenetv2", directory=save_ckpt_path, config=config_ck)

    print("============== Starting Training ==============")
    start_time = time.time()
    # begin train
    cb += [ckpt_cb]
    model.train(config_cpu.epoch_size,
                train_dataset,
                callbacks=cb,
                dataset_sink_mode=False)
    print("time: ", (time.time() - start_time) * 1000)
    print("============== Train Success ==============")
