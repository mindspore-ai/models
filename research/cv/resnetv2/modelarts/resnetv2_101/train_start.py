# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
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
"""train resnetv2."""

import argparse
import os
import numpy as np

from mindspore.nn import Momentum
from mindspore import Model, Tensor, load_checkpoint, load_param_into_net, export, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
# should find /src

from src.lr_generator import get_lr

parser = argparse.ArgumentParser('mindspore resnetv2 training')

parser.add_argument('--net', type=str, default='resnetv2_50',
                    help='Resnetv2 Model, resnetv2_50, resnetv2_101, resnetv2_152')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Dataset, cifar10, imagenet2012')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--train_url', type=str, required=True, default='',
                    help='where training ckpts saved')
parser.add_argument('--data_url', type=str, required=True, default='',
                    help='path of dataset')

parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')

# train
parser.add_argument('--pre_trained', type=str, default=None, help='pretrained checkpoint path')
parser.add_argument('--epoch_size', type=int, default=None, help='epochs')
parser.add_argument('--lr_init', type=float, default=None, help='base learning rate')

# export
parser.add_argument('--width', type=int, default=32, help='input width')
parser.add_argument('--height', type=int, default=32, help='input height')
parser.add_argument('--file_name', type=str, default='resnetv2', help='output air file name')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")

args, _ = parser.parse_known_args()

# import net
if args.net == "resnetv2_50":
    from src.resnetv2 import PreActResNet50 as resnetv2
elif args.net == 'resnetv2_101':
    from src.resnetv2 import PreActResNet101 as resnetv2
elif args.net == 'resnetv2_152':
    from src.resnetv2 import PreActResNet152 as resnetv2
else:
    raise ValueError("network is not support.")

# import dataset and config
if args.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
    from src.config import config1 as config
elif args.dataset == "cifar100":
    from src.dataset import create_dataset2 as create_dataset
    from src.config import config2 as config
elif args.dataset == 'imagenet2012':
    from src.dataset import create_dataset3 as create_dataset
    from src.config import config3 as config
else:
    raise ValueError("dataset is not support.")

def _train():
    """ train """
    print("============== Starting Training ==============")
    target = args.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    if args.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        # init parallel training parameters
        context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        # init HCCL
        init()

    # create dataset
    dataset = create_dataset(dataset_path=args.data_url, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target, distribute=args.run_distribute)

    step_size = dataset.get_dataset_size()

    # define net
    epoch_size = args.epoch_size if args.epoch_size else config.epoch_size
    net = resnetv2(config.class_num, config.low_memory)

    # init weight
    if args.pre_trained:
        param_dict = load_checkpoint(args.pre_trained)
        load_param_into_net(net, param_dict)

    # init lr
    lr_init = args.lr_init if args.lr_init else config.lr_init
    lr = get_lr(lr_init=lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define loss, opt, model
    if args.dataset == "imagenet2012":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                   config.weight_decay, config.loss_scale)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()

    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_save_dir = args.train_url if args.train_url else config.save_checkpoint_path
    ckpoint_cb = ModelCheckpoint(prefix=f"train_{args.net}_{args.dataset}",
                                 directory=ckpt_save_dir, config=config_ck)

    # train
    if args.run_distribute:
        callbacks = [time_cb, loss_cb]
        if target == "GPU" and str(get_rank()) == '0':
            callbacks = [time_cb, loss_cb, ckpoint_cb]
        elif target == "Ascend" and device_id == 0:
            callbacks = [time_cb, loss_cb, ckpoint_cb]
    else:
        callbacks = [time_cb, loss_cb, ckpoint_cb]
    model.train(epoch_size, dataset, callbacks=callbacks)

def _get_last_ckpt(ckpt_dir):
    """ get ckpt """
    ckpt_files = [(os.stat(os.path.join(ckpt_dir, ckpt_file)).st_ctime, ckpt_file)
                  for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, max(ckpt_files)[1])

def _export_air():
    """ export air """
    print("============== Starting Exporting ==============")
    ckpt_file = _get_last_ckpt(args.train_url)
    if not ckpt_file:
        return

    net = resnetv2(config.class_num)
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([config.batch_size, 3, args.height, args.width], np.float32))
    export(net, input_arr, file_name=os.path.join(args.train_url, args.file_name), file_format=args.file_format)

if __name__ == '__main__':
    set_seed(1)
    _train()
    _export_air()
