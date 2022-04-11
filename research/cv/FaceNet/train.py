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
"""Train"""

import os
import argparse

from src.models import FaceNetModelwithLoss
from src.config import facenet_cfg
from src.data_loader import get_dataloader

import mindspore.nn as nn
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import context
from mindspore.common import set_seed
set_seed(0)



parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument("--data_url", type=str, default="/data1/face/FaceNet_mindspore/vggface2/")
parser.add_argument("--train_url", type=str, default="/data1/face/FaceNet_mindspore_final/result/")
parser.add_argument("--data_triplets", type=str, default="/data1/face/FaceNet_mindspore/triplets.csv")
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--run_online", type=str, default='False')
parser.add_argument("--is_distributed", type=str, default='False')
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--group_size", type=int, default=1)

args = parser.parse_args()



class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value



def main():
    cfg = facenet_cfg
    device_num = int(os.environ.get("RANK_SIZE", 1))
    if device_num == 1:
        args.is_distributed = 'False'
    elif device_num > 1:
        args.is_distributed = 'True'

    if args.is_distributed == 'True':
        print("parallel init", flush=True)
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
        context.reset_auto_parallel_context()
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)
        context.set_auto_parallel_context(parameter_broadcast=True)

    if args.run_online == 'True':
        import moxing as mox
        local_data_url = '/cache/data/'
        mox.file.copy_parallel(args.data_url, local_data_url)
        local_train_url = "/cache/train_out/"
    else:
        local_data_url = args.data_url
        local_train_url = args.train_url
        local_triplets = args.data_triplets

    train_root_dir = local_data_url
    valid_root_dir = local_data_url
    train_triplets = local_triplets
    valid_triplets = local_triplets

    ckpt_path = local_train_url


    net = FaceNetModelwithLoss(num_classes=500, margin=cfg.margin, mode='train')


    optimizer = nn.Adam(net.trainable_params(), learning_rate=cfg.learning_rate)

    data_loaders, _ = get_dataloader(train_root_dir, valid_root_dir, train_triplets, valid_triplets,
                                     cfg.batch_size, cfg.num_workers, args.group_size, args.rank,
                                     shuffle=True, mode=args.mode)
    data_loader = data_loaders['train']


    loss_cb = LossMonitor(per_print_times=cfg.per_print_times)
    time_cb = TimeMonitor(data_size=cfg.per_print_times)

    # checkpoint save
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.per_print_times, keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(f"facenet-rank{args.rank}", ckpt_path + 'rank_' + str(args.rank), config_ck)

    callbacks = [loss_cb, time_cb, ckpoint_cb]

    model = Model(net, optimizer=optimizer)


    print("============== Starting Training ==============")
    model.train(cfg.num_epochs, data_loader, callbacks=callbacks, dataset_sink_mode=True)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', save_graphs=False)
    main()
