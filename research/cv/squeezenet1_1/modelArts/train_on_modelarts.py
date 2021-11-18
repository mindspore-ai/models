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
"""train squeezenet."""
import ast
import os
import argparse
import glob
import numpy as np

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
from mindspore.common import set_seed
from mindspore.nn.metrics import Accuracy
from mindspore.communication.management import init
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.squeezenet import SqueezeNet as squeezenet

parser = argparse.ArgumentParser(description='SqueezeNet1_1')
parser.add_argument('--net', type=str, default='squeezenet', help='Model.')
parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset.')
parser.add_argument('--run_cloudbrain', type=ast.literal_eval, default=False,
                    help='Whether it is running on CloudBrain platform.')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--pre_trained', type=str, default="None", help='Pretrained checkpoint path')
parser.add_argument('--data_url', type=str, default="None", help='Datapath')
parser.add_argument('--train_url', type=str, default="None", help='Train output path')
parser.add_argument('--num_classes', type=int, default="1000", help="classes")
parser.add_argument('--epoch_size', type=int, default="200", help="epoch_size")
parser.add_argument('--batch_size', type=int, default="32", help="batch_size")
args_opt = parser.parse_args()

local_data_url = '/cache/data'
local_train_url = '/cache/ckpt'
local_pretrain_url = '/cache/preckpt.ckpt'

set_seed(1)

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def frozen_to_air(network, args):
    paramdict = load_checkpoint(args.get("ckpt_file"))
    load_param_into_net(network, paramdict)
    input_arr = Tensor(np.zeros([args.get("batch_size"), 3, args.get("height"), args.get("width")], np.float32))
    export(network, input_arr, file_name=args.get("file_name"), file_format=args.get("file_format"))

if __name__ == '__main__':

    target = args_opt.device_target
    if args_opt.device_target != "Ascend":
        raise ValueError("Unsupported device target.")

    # init context
    if args_opt.run_distribute:
        device_num = int(os.getenv("RANK_SIZE"))
        device_id = int(os.getenv("DEVICE_ID"))
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=target)
        context.set_context(device_id=device_id,
                            enable_auto_mixed_precision=True)
        context.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True)
        init()
        local_data_url = os.path.join(local_data_url, str(device_id))

    else:
        device_id = 0
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=target)

    # create dataset
    if args_opt.dataset == "cifar10":
        from src.config import config_cifar as config
        from src.dataset import create_dataset_cifar as create_dataset
    else:
        from src.config import config_imagenet as config
        from src.dataset import create_dataset_imagenet as create_dataset

    if args_opt.run_cloudbrain:
        import moxing as mox
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        dataset = create_dataset(dataset_path=local_data_url,
                                 do_train=True,
                                 repeat_num=1,
                                 batch_size=args_opt.batch_size,
                                 target=target,
                                 run_distribute=args_opt.run_distribute)


    step_size = dataset.get_dataset_size()

    # define net
    net = squeezenet(num_classes=args_opt.num_classes)

    # load checkpoint
    if args_opt.pre_trained != "None":
        if args_opt.run_cloudbrain:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            ckpt_name = args_opt.pre_trained[2:]
            ckpt_path = os.path.join(dir_path, ckpt_name)
            print(ckpt_path)
            param_dict = load_checkpoint(ckpt_path)
            filter_list = [x.name for x in net.final_conv.get_parameters()]
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
            load_param_into_net(net, param_dict)


    # init lr
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                total_epochs=args_opt.epoch_size,
                warmup_epochs=config.warmup_epochs,
                pretrain_epochs=config.pretrain_epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define loss
    if args_opt.dataset == "imagenet":
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True,
                                  reduction='mean',
                                  smooth_factor=config.label_smooth_factor,
                                  num_classes=args_opt.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define opt, model
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
                  metrics={'acc': Accuracy()},
                  amp_level="O2",
                  keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint and device_id == 0:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
            keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=args_opt.net,
                                  directory=local_train_url,
                                  config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(args_opt.epoch_size - config.pretrain_epoch_size,
                dataset,
                callbacks=cb)
    if device_id == 0:
        ckpt_list = glob.glob("/cache/ckpt/squeezenet*.ckpt")

        if not ckpt_list:
            print("ckpt file not generated.")

        ckpt_list.sort(key=os.path.getmtime)
        ckpt_model = ckpt_list[-1]
        print("checkpoint path", ckpt_model)

        net = squeezenet(args_opt.num_classes)

        frozen_to_air_args = {'ckpt_file': ckpt_model,
                              'batch_size': 1,
                              'height': 227,
                              'width': 227,
                              'file_name': '/cache/ckpt/squeezenet',
                              'file_format': 'AIR'}
        frozen_to_air(net, frozen_to_air_args)

    if args_opt.run_cloudbrain:
        mox.file.copy_parallel(local_train_url, args_opt.train_url)
