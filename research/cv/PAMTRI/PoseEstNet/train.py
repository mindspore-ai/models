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
"""
########################## train PoseEstNet ##########################
train PoseEstNet and get network model files(.ckpt) :
python train.py --cfg config.yaml --pre_ckpt_path pretrained.ckpt --data_dir datapath
"""
import os
import ast
import argparse
import mindspore
import mindspore.nn as nn

from mindspore import Tensor
from mindspore.common import set_seed
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.scheduler import get_lr
from src.loss import NetWithLoss
from src.model import get_pose_net
from src.dataset import create_dataset
from src.config import cfg, update_config

parser = argparse.ArgumentParser(description='Train PoseEstNet')

parser.add_argument('--train_url', type=str)
parser.add_argument('--data_url', type=str)
parser.add_argument('--isModelArts', type=ast.literal_eval, default=False)
parser.add_argument('--cfg', required=True, type=str)
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--pre_trained', type=str, default=True)
parser.add_argument('--pre_ckpt_path', type=str, default='')
parser.add_argument('--device_target', type=str, default="Ascend")
parser.add_argument('--distribute', type=ast.literal_eval, default=True)

args = parser.parse_args()

if args.isModelArts:
    import moxing as mox

if __name__ == '__main__':
    set_seed(1)
    update_config(cfg, args)

    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args.distribute:
        init()
        device_num = int(os.getenv('RANK_SIZE', '1'))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, device_num=device_num)
    else:
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)


    #define dataset
    if args.isModelArts:
        mox.file.copy_parallel(src_url=args.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID')
        dataset = create_dataset(cfg, train_dataset_path, is_train=True)
    else:
        dataset = create_dataset(cfg, args.data_dir, is_train=True)

    step_size = dataset.get_dataset_size()

    #define net
    network = get_pose_net(cfg)

    if args.pre_trained:
        if args.isModelArts:
            pre_path = train_dataset_path + '/' + 'hrnet.ckpt'
            param_dict = load_checkpoint(pre_path)
            load_param_into_net(network, param_dict)
            print("pre_trained is done")
        else:
            param_dict = load_checkpoint(args.pre_ckpt_path)
            load_param_into_net(network, param_dict)
            print("pre_trained is done")

    net_with_loss = NetWithLoss(network, use_target_weight=True)

    #init lr
    lr = get_lr(lr=cfg.TRAIN.LR,
                total_epochs=cfg.TRAIN.END_EPOCH,
                steps_per_epoch=step_size,
                lr_step=cfg.TRAIN.LR_STEP,
                gamma=cfg.TRAIN.LR_FACTOR)
    lr = Tensor(lr, mindspore.float32)

    #define opt
    decayed_params = []
    no_decayed_params = []
    for param in network.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': cfg.TRAIN.WD},
                    {'params': no_decayed_params},
                    {'order_params': network.trainable_params()}]

    optimizer = nn.Adam(group_params,
                        learning_rate=lr,
                        weight_decay=cfg.TRAIN.WD,
                        use_nesterov=cfg.TRAIN.NESTEROV)

    model = Model(net_with_loss, optimizer=optimizer)

    # define callbacks
    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]

    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.TRAIN.END_EPOCH * step_size,
                                 keep_checkpoint_max=cfg.TRAIN.END_EPOCH)

    if args.isModelArts:
        save_checkpoint_path = '/cache/train_output/'
    elif not args.distribute:
        save_checkpoint_path = './ckpt/'
    else:
        save_checkpoint_path = './ckpt_' + str(get_rank()) + '/'

    ckpt_cb = ModelCheckpoint(prefix='PoseEstNet',
                              directory=save_checkpoint_path,
                              config=config_ck)
    cb += [ckpt_cb]

    print("===================================")
    print("Total epoch: {}".format(cfg.TRAIN.END_EPOCH))
    print("Batch size: {}".format(cfg.TRAIN.BATCH_SIZE))
    print("==========Training begin===========")
    model.train(cfg.TRAIN.END_EPOCH, dataset, callbacks=cb, dataset_sink_mode=True)
    if args.isModelArts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args.train_url)
