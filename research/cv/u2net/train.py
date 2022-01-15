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
"""train process"""

import argparse
import ast
import os
import random

import numpy as np
from mindspore import FixedLossScaleManager
from mindspore import Model
from mindspore import context
from mindspore import dataset as ds
from mindspore import nn
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import src.config as config
from src.blocks import U2NET
from src.data_loader import create_dataset
from src.loss import total_loss

random.seed(1)
np.random.seed(1)
ds.config.set_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--content_path", type=str, help='content_path, default: None')
parser.add_argument("--label_path", type=str, help='label_path, default: None')
parser.add_argument('--ckpt_path', type=str, default='ckpts', help='checkpoint save location, default: ckpts')
parser.add_argument("--ckpt_name", default='u2net', type=str, help='prefix of ckpt files, default: u2net')
parser.add_argument("--loss_scale", type=int, default=8192)
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute, default: false.")
parser.add_argument('--pre_trained', default='', type=str, help='model_path, local pretrained model to load')
parser.add_argument('--device_target', type=str, default="Ascend", choices=("Ascend", "GPU"),
                    help="Device target, support GPU and CPU.")
# additional params for online training
parser.add_argument("--run_online", type=int, default=0, help='whether train online, default: false')
parser.add_argument("--data_url", type=str, help='path to data on obs, default: None')
parser.add_argument("--train_url", type=str, help='output path on obs, default: None')
parser.add_argument("--is_load_pre", type=int, default=0, help="whether use pretrained model, default: false.")

args = parser.parse_args()

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)


    if args.run_distribute:
        init()
        if args.device_target == "Ascend":
            cfg = config.run_distribute_cfg
            device_id = int(os.getenv('DEVICE_ID'))
        else:
            cfg = config.run_distribute_cfg_GPU
            device_id = get_rank()
            args.ckpt_path = os.path.join(args.ckpt_path, str(device_id))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          device_num=device_num)

    else:
        if args.device_target == "Ascend":
            cfg = config.single_cfg
            device_id = int(os.getenv('DEVICE_ID'))
        else:
            cfg = config.single_cfg_GPU

    if args.run_online:
        import moxing as mox

        mox.file.copy_parallel(args.data_url, "/cache/dataset")
        content_path = "/cache/dataset/DUTS/DUTS-TR/DUTS-TR-Image/"
        label_path = "/cache/dataset/DUTS/DUTS-TR/DUTS-TR-Mask/"
        if args.run_distribute:
            args.ckpt_path = "/cache/ckpts/device" + str(device_id)
        else:
            args.ckpt_path = "/cache/ckpts"
        if args.is_load_pre:
            pre_ckpt_dir = "/cache/dataset/pre_ckpt"
            args.pre_trained = pre_ckpt_dir + "/" + os.listdir(pre_ckpt_dir)[0]
    else:
        content_path = args.content_path
        label_path = args.label_path


    args.lr = cfg.lr
    args.batch_size = cfg.batch_size
    args.decay = cfg.weight_decay
    args.epoch_size = cfg.max_epoch
    args.eps = cfg.eps
    print("---")
    print("eps = %lf,batch_size = %d, epoch size = %d" % (args.eps, args.batch_size, args.epoch_size))
    print("lr = %lf,decay = %f, ckpt_name = %s, loss_scale = %d"
          % (args.lr, args.decay, args.ckpt_name, args.loss_scale))
    net = U2NET()
    net.set_train()
    if args.pre_trained != '':
        print("pretrained path = %s" % args.pre_trained)
        param_dict = load_checkpoint(args.pre_trained)
        load_param_into_net(net, param_dict)
    print("---")
    loss = total_loss()
    ds_train = create_dataset(content_path, label_path, args)
    print("dataset size: ", ds_train.get_dataset_size())
    opt = nn.Adam(net.get_parameters(), learning_rate=args.lr, beta1=0.9, beta2=0.999, eps=args.eps,
                  weight_decay=args.decay)
    loss_scale_manager = FixedLossScaleManager(args.loss_scale)
    model = Model(net, loss, opt, loss_scale_manager=loss_scale_manager, amp_level="O0")
    data_size = ds_train.get_dataset_size()
    config_ck = CheckpointConfig(save_checkpoint_steps=data_size, keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=args.ckpt_name, directory=args.ckpt_path, config=config_ck)
    net.set_train()
    model.train(args.epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(per_print_times=1), TimeMonitor()])
    if args.run_online:
        mox.file.copy_parallel(args.ckpt_path, args.train_url)
