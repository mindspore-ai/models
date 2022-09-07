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

"""train_criteo."""
import argparse
import json
import os

from mindspore import context, Tensor, ParameterTuple
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.nn.optim import AdamWeightDecay
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore as ms
import mindspore.nn as nn

from src.callback import TimeMonitor, Monitor
from src.config import train_config, symbols, encoder_kw, decoder_kw
from src.dataset import create_train_dataset

from src.model import Jasper, NetWithLossClass, init_weights
from src.eval_callback import SaveCallback
from src.lr_generator import get_lr

parser = argparse.ArgumentParser(description='Jasper training')
parser.add_argument('--pre_trained_model_path', type=str,
                    default='', help='Pretrained checkpoint path')
parser.add_argument('--is_distributed', action="store_true",
                    default=False, help='Distributed training')
parser.add_argument('--device_target', type=str, default="GPU", choices=("GPU", "CPU", "Ascend"),
                    help='Device target, support GPU and CPU, Default: GPU')
parser.add_argument('--device_id', type=int, default=0, help='Device ID')
args = parser.parse_args()

ms.set_seed(1)

if __name__ == '__main__':

    rank_id = 0
    group_size = 1
    config = train_config
    data_sink = False
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target, save_graphs=False)

    if args.is_distributed:
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            device_num=get_group_size(),
            gradients_mean=True)
        rank_id = get_rank()
        group_size = get_group_size()
    else:
        if args.device_target == "Ascend":
            device_id = int(args.device_id)
            context.set_context(device_id=device_id)

    with open(config.DataConfig.labels_path) as label_file:
        labels = json.load(label_file)
    bs = config.DataConfig.batch_size
    ds_train = create_train_dataset(mindrecord_files=config.DataConfig.mindrecord_files,
                                    labels=symbols, batch_size=bs, train_mode=True,
                                    rank=rank_id, group_size=group_size)
    steps_size = ds_train.get_dataset_size()

    lr = get_lr(lr_init=config.OptimConfig.learning_rate, total_epochs=config.TrainingConfig.epochs,
                steps_per_epoch=steps_size)
    lr = Tensor(lr)

    jasper_net = Jasper(encoder_kw=encoder_kw,
                        decoder_kw=decoder_kw).to_float(ms.float16)

    loss_net = NetWithLossClass(jasper_net, ascend=(args.device_target == "Ascend"))
    init_weights(loss_net)
    weights = ParameterTuple(jasper_net.trainable_params())
    optimizer = AdamWeightDecay(weights, learning_rate=lr, eps=config.OptimConfig.epsilon, weight_decay=1e-3)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    train_net.set_train(True)
    if args.pre_trained_model_path != '':
        param_dict = load_checkpoint(args.pre_trained_model_path)
        load_param_into_net(loss_net, param_dict)
        print('Successfully loading the pre-trained model')

    loss_scale = 128.0
    loss_scale = FixedLossScaleManager(loss_scale, drop_overflow_update=True)
    model = Model(loss_net, optimizer=optimizer, loss_scale_manager=loss_scale)

    callback_list = [TimeMonitor(steps_size), Monitor(lr)]

    if args.is_distributed:
        print('Distributed training.')
        config.CheckpointConfig.ckpt_path = os.path.join(config.CheckpointConfig.ckpt_path,
                                                         'ckpt_' + str(get_rank()) + '/')
        if rank_id == 0:
            callback_update = SaveCallback(config.CheckpointConfig.ckpt_path)
            callback_list += [callback_update]
    else:
        print('Standalone training.')
        config_ck = CheckpointConfig(save_checkpoint_steps=1000,
                                     keep_checkpoint_max=config.CheckpointConfig.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=config.CheckpointConfig.ckpt_file_name_prefix,
                                  directory=config.CheckpointConfig.ckpt_path, config=config_ck)

        callback_list.append(ckpt_cb)
    model.train(config.TrainingConfig.epochs, ds_train,
                callbacks=callback_list, dataset_sink_mode=data_sink)
