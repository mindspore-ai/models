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



import ast
import argparse
import os
import numpy as np

from mindspore import context, export
from mindspore import Tensor
from mindspore.nn import SGD, RMSProp
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
import moxing as mox
from src.lr_generator import get_lr
from src.models.effnet import EfficientNet
from src.config import config
from src.monitor import Monitor
from src.dataset import create_dataset
from src.loss import CrossEntropySmooth

set_seed(1)


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key, flush=True)
                del origin_dict[key]
                break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='image classification training')
    # modelarts parameter
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')

    # Ascend parameter
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='Run distribute')
    parser.add_argument('--device_num', type=int, default=0, help='Device num')
    parser.add_argument('--device_id', type=int, default=0, help='Device id')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')

    parser.add_argument('--run_modelarts', type=ast.literal_eval, default=True, help='Run mode')
    parser.add_argument('--ckpt_path', type=str, default=None, help='ckpt_path training with existed checkpoint')
    parser.add_argument('--trans_learning', type=ast.literal_eval, default=False)
    args_opt = parser.parse_args()


    if args_opt.run_modelarts:
        local_data_url = "/cache/data"
        local_train_url = "/cache/ckpt"
        local_ckpt_url = "/cache/pretrained.ckpt" if args_opt.ckpt_path else ""
        mox.file.copy_parallel(args_opt.data_url, local_data_url)

        if local_ckpt_url:
            mox.file.copy_parallel(args_opt.ckpt_path, local_ckpt_url)

    else:
        local_data_url = args_opt.dataset_path
        local_train_url = config.save_checkpoint_path
        local_ckpt_url = args_opt.ckpt_path

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

    if args_opt.run_distribute:
        init()
        args_opt.device_id = int(os.getenv('DEVICE_ID'))
        args_opt.device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=args_opt.device_id)
        context.set_auto_parallel_context(device_num=args_opt.device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        args_opt.device_id = 0
        args_opt.device_num = 1

    net = EfficientNet(1, 1)
    net.to_float(mstype.float16)

    # if args_opt.ckpt_path:
    if local_ckpt_url:
        ckpt = load_checkpoint(local_ckpt_url)
        if args_opt.trans_learning:
            filter_list = [x.name for x in net.head.get_parameters()]
            filter_checkpoint_parameter_by_list(ckpt, filter_list)
        load_param_into_net(net, ckpt)
    net.set_train(True)

    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    dataset = create_dataset(dataset_path=local_data_url,
                             do_train=True,
                             batch_size=config.batch_size,
                             device_num=args_opt.device_num, rank=args_opt.device_id)
    step_size = dataset.get_dataset_size()  # 计算每epoch有多少step

    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size,
                       lr_decay_mode=config.lr_decay_mode))

    if config.opt == 'sgd':
        optimizer = SGD(net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                        weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    elif config.opt == 'rmsprop':
        optimizer = RMSProp(net.trainable_params(), learning_rate=lr, decay=0.9, weight_decay=config.weight_decay,
                            momentum=config.momentum, epsilon=config.opt_eps, loss_scale=config.loss_scale)
    else:
        raise ValueError("Unsupported optimizer.")

    model = Model(net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale,
                  metrics={'acc'}, amp_level='O3')

    # define callbacks
    cb = [Monitor(lr_init=lr.asnumpy())]
    if config.save_checkpoint and args_opt.device_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(f"Efficientnet_b0-rank{args_opt.device_id}", \
                                  directory=local_train_url, \
                                  config=config_ck)
        cb += [ckpt_cb]

    model.train(config.epoch_size, dataset, callbacks=cb)

    if args_opt.device_id == 0:
        net.set_train(False)
        input_shp = [1, 3, 224, 224]
        input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
        file_name = os.path.join(local_train_url, "efficient-net-b0")
        export(net, input_array, file_name=file_name, file_format="AIR")

    if args_opt.run_modelarts and args_opt.device_id == 0:
        mox.file.copy_parallel(local_train_url, args_opt.train_url)
