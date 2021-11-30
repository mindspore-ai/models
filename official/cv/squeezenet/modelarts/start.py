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
"""train squeezenet."""
import os
import sys
import argparse
import numpy as np
import moxing as mox

from mindspore import context
from mindspore import Tensor
from mindspore import export
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth


set_seed(1)

_CACHE_DATA_URL = "./cache/data"
_CACHE_TRAIN_URL = "./cache/train"
_CACHE_LOAD_URL = "./cache/checkpoint_path"
_CACHE_PRETRAIN_URL = "./cache/res/tmp.ckpt"
_NONE = "none"


# Transfer learning parameter
parser = argparse.ArgumentParser('mindspore squeezenet_residual training')
parser.add_argument('--train_url', type=str, default='obs://neu-base/squeezenet/cache/train',
                    help='where training log and ckpts saved')
parser.add_argument('--data_url', type=str, default='obs://neu-base/squeezenet/cache/data',
                    help='path of dataset')
parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset.')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--file_format', type=str, default="AIR",
                    help='output model formats')
parser.add_argument('--pre_trained', type=str,
                    default='obs://neu-base/squeezenet/suqeezenet_residual_imagenet-50_3.ckpt',
                    help='pretrained model')
parser.add_argument('--epoch_size', type=int, default=1,
                    help='epoch num')
parser.add_argument('--save_checkpoint_epochs', type=int, default=1,
                    help='how many epochs to save ckpt once')
parser.add_argument('--device_target', type=str, default='Ascend',
                    choices=['Ascend', 'CPU', 'GPU'],
                    help='device where the code will be implemented. '
                    '(Default: Ascend)')
parser.add_argument('--file_name', type=str, default="squeezenet_residual",
                    help='output file name')
parser.add_argument('--lr_max', type=float, default=0.01,
                    help='lr_max')
args, _ = parser.parse_known_args()

os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
os.makedirs(_CACHE_DATA_URL, exist_ok=True)
os.makedirs(_CACHE_LOAD_URL, exist_ok=True)
mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)
train_url = _CACHE_TRAIN_URL
data_url = _CACHE_DATA_URL
load_url = _CACHE_LOAD_URL

if args.dataset == "cifar10":
    from src.config import config_cifar as config
else:
    from src.config import config_imagenet as config
print("Dataset: ", config.dataset)


#train
if config.net_name == "squeezenet":
    from src.squeezenet import SqueezeNet as squeezenet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.dataset import create_dataset_imagenet as create_dataset
else:
    from src.squeezenet import SqueezeNet_Residual as squeezenet
    if config.dataset == "cifar10":
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.dataset import create_dataset_imagenet as create_dataset

@moxing_wrapper()
def train_net():
    """train net"""
    config.output_path = train_url
    config.data_path = data_url
    config.load_path = load_url
    config.epoch_size = args.epoch_size
    config.save_checkpoint_epochs = args.save_checkpoint_epochs
    config.device_target = args.device_target
    target = config.device_target
    ckpt_save_dir = config.output_path

    if args.pre_trained != _NONE:
        mox.file.copy_parallel(args.pre_trained, _CACHE_PRETRAIN_URL)
    else:
        config.pre_trained = ""

    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target)
    device_num = 1
    if config.run_distribute:
        if target == "Ascend":
            device_id = get_device_id()
            device_num = config.device_num
            context.set_context(device_id=device_id)
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
            init()
        # GPU target
        else:
            init()
            device_num = get_group_size()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
        ckpt_save_dir = ckpt_save_dir + "/ckpt_" + str(
            get_rank()) + "/"

    # create dataset
    dataset = create_dataset(dataset_path=config.data_path,
                             do_train=True,
                             repeat_num=1,
                             batch_size=config.batch_size,
                             run_distribute=config.run_distribute)

    step_size = dataset.get_dataset_size()

    # define net
    net = squeezenet(num_classes=config.class_num)

    # load checkpoint
    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained)
        load_param_into_net(net, param_dict)

    # init lr
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                total_epochs=args.epoch_size,
                warmup_epochs=config.warmup_epochs,
                pretrain_epochs=config.pretrain_epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define loss
    if config.dataset == "imagenet":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True,
                                  reduction='mean',
                                  smooth_factor=config.label_smooth_factor,
                                  num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define opt, model
    if target == "Ascend":
        loss_scale = FixedLossScaleManager(config.loss_scale,
                                           drop_overflow_update=False)
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                       lr,
                       config.momentum,
                       config.weight_decay,
                       config.loss_scale,
                       use_nesterov=True)
        model = Model(net,
                      loss_fn=loss,
                      optimizer=opt,
                      loss_scale_manager=loss_scale,
                      metrics={'acc'},
                      amp_level="O2",
                      keep_batchnorm_fp32=False)
    else:
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                       lr,
                       config.momentum,
                       config.weight_decay,
                       use_nesterov=True)
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
            keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=config.net_name + '_' + config.dataset,
                                  directory=ckpt_save_dir,
                                  config=config_ck)
        cb += [ckpt_cb]
    # train model
    model.train(config.epoch_size - config.pretrain_epoch_size,
                dataset,
                callbacks=cb)

if config.net_name == "squeezenet":
    from src.squeezenet import SqueezeNet as squeezenet
else:
    from src.squeezenet import SqueezeNet_Residual as squeezenet
if config.dataset == "cifar10":
    num_classes = 10
else:
    num_classes = 1000

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    #get_last_ckpt
    ckpt_dir = train_url
    file_dict = {}
    lists = os.listdir(ckpt_dir)
    for i in lists:
        ctime = os.stat(os.path.join(ckpt_dir, i)).st_ctime
        file_dict[ctime] = i
    max_ctime = max(file_dict.keys())
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith(file_dict[max_ctime])]

    if not ckpt_files:
        print("No ckpt file found.")
        sys.exit(0)

    ckpt_file = os.path.join(ckpt_dir, sorted(ckpt_files)[-1])

    # export_air
    if not ckpt_file:
        sys.exit(0)
    config.batch_size = args.batch_size
    config.checkpoint_file_path = ckpt_file
    config.file_name = os.path.join(_CACHE_TRAIN_URL, args.file_name)
    print("\nStart exporting AIR.")

    net = squeezenet(num_classes=num_classes)

    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(net, param_dict)

    input_data = Tensor(np.zeros([config.batch_size, 3, config.height, config.width], np.float32))
    export(net, input_data, file_name=config.file_name, file_format=config.file_format)
    print("\nStart uploading to obs.")
    mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)

if __name__ == '__main__':
    train_net()
    run_export()
