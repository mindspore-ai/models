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

import os
import argparse
import glob
from ast import literal_eval as liter
from mindspore import Tensor
from mindspore import context
from mindspore import export
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.nn.optim import Adam, Momentum
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.openposenet import OpenPoseNet
from src.loss import openpose_loss, BuildTrainNetwork, TrainOneStepWithClipGradientCell
from src.utils import get_lr, MyLossMonitor
from src.model_utils.config import config
from src.model_utils.device_adapter import get_rank_id, get_device_num
import moxing as mox
import numpy as np

parser = argparse.ArgumentParser(description='Object detection')
parser.add_argument('--data_url',
                    metavar='DIR',
                    default='/cache/data_url',
                    help='path to dataset')
parser.add_argument('--train_url',
                    default="/cache/output/",
                    type=str,
                    help="setting dir of training output")
parser.add_argument('--vgg_path',
                    default="",
                    type=str,
                    help="vgg path")
parser.add_argument('--imgpath_train',
                    default="imgpath/train2017",
                    type=str,
                    help="image path")
parser.add_argument('--jsonpath_train',
                    default="jsonpath/person_keypoints_train2017.json",
                    type=str,
                    help="json path")
parser.add_argument('--maskpath_train',
                    default="maskpath/ignore_train2017",
                    type=str,
                    help="mask path")
parser.add_argument('--max_epoch_train',
                    default="2",
                    type=int,
                    help="max_epoch_train")
parser.add_argument('--warmup_epoch',
                    default="1",
                    type=int,
                    help="warmup_epoch")
parser.add_argument('--batch_size',
                    default="128",
                    type=int,
                    help="batch_size")

args_opt = parser.parse_args()

set_seed(1)


def frozen_to_air(net_f, args):
    param_dict = load_checkpoint(args.get("ckpt_file"))
    load_param_into_net(net_f, param_dict)
    input_arr = Tensor(np.zeros([args.get("batch_size"), 3, args.get("height"), args.get("width")], np.float32))
    export(net, input_arr, file_name=args.get("file_name"), file_format=args.get("file_format"))


if __name__ == "__main__":
    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path, exist_ok=True)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path, exist_ok=True)
    mox.file.copy_parallel(args_opt.data_url, config.data_path)

    config.imgpath_train = os.path.join("/cache/data/", args_opt.imgpath_train)
    config.jsonpath_train = os.path.join("/cache/data/", args_opt.jsonpath_train)
    config.maskpath_train = os.path.join("/cache/data/", args_opt.maskpath_train)
    config.max_epoch_train = args_opt.max_epoch_train
    config.warmup_epoch = args_opt.warmup_epoch
    config.batch_size = args_opt.batch_size
    if args_opt.vgg_path:
        config.vgg_path = os.path.join("/cache/data/", args_opt.vgg_path)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

    config.lr = liter(config.lr)
    config.outputs_dir = config.save_model_path
    device_num = get_device_num()

    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        config.rank = get_rank_id()
        config.outputs_dir = os.path.join(config.outputs_dir, "ckpt_{}/".format(config.rank))
    else:
        config.outputs_dir = os.path.join(config.outputs_dir, "ckpt_0/")
        config.rank = 0

    if device_num > 1:
        config.max_epoch = config.max_epoch_train_NP
        config.loss_scale = config.loss_scale / 2
        config.lr_steps = list(map(int, config.lr_steps_NP.split(',')))
        config.train_type = config.train_type_NP
        config.optimizer = config.optimizer_NP
        config.group_params = config.group_params_NP
    else:
        config.max_epoch = config.max_epoch_train
        config.loss_scale = config.loss_scale
        config.lr_steps = list(map(int, config.lr_steps.split(',')))

    # create network
    print('start create network')
    criterion = openpose_loss()
    criterion.add_flags_recursive(fp32=True)
    network = OpenPoseNet(vggpath=config.vgg_path)
    train_net = BuildTrainNetwork(network, criterion)

    # create dataset
    if os.path.exists(config.jsonpath_train) and os.path.exists(config.imgpath_train) \
            and os.path.exists(config.maskpath_train):
        print('start create dataset')
    else:
        raise ValueError('Error: wrong data path.')

    num_worker = 20 if device_num > 1 else 48
    de_dataset_train = create_dataset(config.jsonpath_train, config.imgpath_train, config.maskpath_train,
                                      batch_size=config.batch_size,
                                      rank=config.rank,
                                      group_size=device_num,
                                      num_worker=num_worker,
                                      multiprocessing=True,
                                      shuffle=True,
                                      repeat_num=1)
    steps_per_epoch = de_dataset_train.get_dataset_size()
    print("steps_per_epoch: ", steps_per_epoch)

    # lr scheduler
    lr_stage, lr_base, lr_vgg = get_lr(config.lr * device_num,
                                       config.lr_gamma,
                                       steps_per_epoch,
                                       config.max_epoch,
                                       config.lr_steps,
                                       device_num,
                                       lr_type=config.lr_type,
                                       warmup_epoch=config.warmup_epoch)

    # optimizer
    if config.group_params:
        vgg19_base_params = list(filter(lambda x: 'base.vgg_base' in x.name, train_net.trainable_params()))
        base_params = list(filter(lambda x: 'base.conv' in x.name, train_net.trainable_params()))
        stages_params = list(filter(lambda x: 'base' not in x.name, train_net.trainable_params()))

        group_params = [{'params': vgg19_base_params, 'lr': lr_vgg},
                        {'params': base_params, 'lr': lr_base},
                        {'params': stages_params, 'lr': lr_stage}]

        if config.optimizer == "Momentum":
            opt = Momentum(group_params, learning_rate=lr_stage, momentum=0.9)
        elif config.optimizer == "Adam":
            opt = Adam(group_params)
        else:
            raise ValueError("optimizer not support.")
    else:
        if config.optimizer == "Momentum":
            opt = Momentum(train_net.trainable_params(), learning_rate=lr_stage, momentum=0.9)
        elif config.optimizer == "Adam":
            opt = Adam(train_net.trainable_params(), learning_rate=lr_stage)
        else:
            raise ValueError("optimizer not support.")

    # callback
    config_ck = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=f"openpose-train-rank{config.rank}", directory=config.outputs_dir,
                                 config=config_ck)
    time_cb = TimeMonitor(data_size=de_dataset_train.get_dataset_size())
    if config.rank == 0:
        callback_list = [MyLossMonitor(), time_cb, ckpoint_cb]
    else:
        callback_list = [MyLossMonitor(), time_cb]

    # train
    if config.train_type == 'clip_grad':
        train_net = TrainOneStepWithClipGradientCell(train_net, opt, sens=config.loss_scale)
        train_net.set_train()
        model = Model(train_net)
    elif config.train_type == 'fix_loss_scale':
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        train_net.set_train()
        model = Model(train_net, optimizer=opt, loss_scale_manager=loss_scale_manager)
    else:
        raise ValueError("Type {} is not support.".format(config.train_type))

    print("============== Starting Training ==============")
    model.train(config.max_epoch, de_dataset_train, callbacks=callback_list,
                dataset_sink_mode=False)

    ckpt_list = glob.glob(config.outputs_dir + "/openpose*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    net = OpenPoseNet(vggpath=config.vgg_path)

    frozen_to_air_args = {'ckpt_file': ckpt_model,
                          'batch_size': 1,
                          'height': 368,
                          'width': 368,
                          'file_name': config.outputs_dir + '/openpose',
                          'file_format': 'AIR'}
    frozen_to_air(net, frozen_to_air_args)
    mox.file.copy_parallel(config.outputs_dir, args_opt.train_url)
