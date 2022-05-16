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
Use this file for standalone training and distributed training
"""

import argparse
import ast
import os

import mindspore.nn as nn
from mindspore import context, Tensor, export
from mindspore.common import set_seed
import mindspore.dataset as ds

from mindspore.train.model import Model
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn import WithLossCell, TrainOneStepCell
from src.config import CONFIG
from src.dataset import TrainDataset
from src.utils import AverageMeter
from src.ctrl import CTRL, CTRL_Loss
import numpy as np

def get_args():
    """ get args"""
    parser = argparse.ArgumentParser(description='Train CTRL')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend'],
                        help='device target, only support Ascend.')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend.')
    parser.add_argument('--max_epoch', type=int, default=1, help='max train epoch')
    parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR',
                        help='file format')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='run distribute.')
    parser.add_argument('--train_data_dir', type=str, default=None, help='the directory of train data.')
    parser.add_argument('--check_point_dir', type=str, default=None, help='the directory of train check_point.')
    parser.add_argument('--train_url', default=None, help='Cloudbrain Location of training outputs.\
                        This parameter needs to be set when running on the cloud brain platform.')
    parser.add_argument('--data_url', default=None, help='Cloudbrain Location of data.\
                        This parameter needs to be set when running on the cloud brain platform.')
    parser.add_argument('--run_cloudbrain', type=ast.literal_eval, default=False,
                        help='Whether it is running on CloudBrain platform.')
    parser.add_argument('--train_output_path', type=str, default=None)
    return parser.parse_args()


args = get_args()
local_data_url = './cache/data'
local_train_url = './cache/train'
_local_train_url = local_train_url
if args.run_cloudbrain:
    import moxing as mox

    args.train_data_dir = local_data_url
    device_id = int(os.getenv('DEVICE_ID'))
    args.train_output_path = os.path.join(local_train_url, f"logs_{int(device_id)}")
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
else:
    device_id = args.device_id
cfg = CONFIG(data_dir=args.train_data_dir, log_dir=args.train_output_path)
cfg.max_epoch = args.max_epoch
set_seed(cfg.seed)
if __name__ == '__main__':
    print("Set Context...")
    rank_size = int(os.getenv('RANK_SIZE')) if args.run_distribute else 1
    rank_id = int(os.getenv('RANK_ID')) if args.run_distribute else 0
    print(f"device_id:{device_id}, rank_id:{rank_id}")
    print(f"args.device_id:{args.device_id}")
    context.set_context(mode=cfg.mode, device_target=args.device_target,
                        device_id=device_id, save_graphs=False)
    if args.run_distribute:
        print("Init distribute train...")
        cfg.batch_size = 8
        cfg.max_epoch = 10
        cfg.optimizer = 'Momentum'
        init()
        context.set_auto_parallel_context(device_num=rank_size,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    print('Done.')
    print("Get Dataset...")

    dataset = TrainDataset(cfg.train_feature_dir, cfg.train_csv_path, cfg.valid_csv_path,
                           cfg.visual_dim, cfg.sentence_embed_dim, cfg.IoU, cfg.nIoL,
                           cfg.context_num, cfg.context_size)
    if args.run_distribute:
        dataset = ds.GeneratorDataset(dataset, ["vis_sent", "offset"], shuffle=False,
                                      num_shards=rank_size, shard_id=rank_id)
    else:
        dataset = ds.GeneratorDataset(dataset, ["vis_sent", "offset"], shuffle=False)
    dataset = dataset.shuffle(buffer_size=cfg.buffer_size)
    dataset = dataset.batch(batch_size=cfg.batch_size)
    print('Done.')

    print("Get Model...")
    net = CTRL(cfg.visual_dim, cfg.sentence_embed_dim, cfg.semantic_dim, cfg.middle_layer_dim)
    loss = CTRL_Loss(cfg.lambda_reg)


    if cfg.optimizer == 'Adam':
        net_opt = nn.Adam(net.trainable_params(), learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Momentum':
        net_opt = nn.Momentum(net.trainable_params(), cfg.lr, cfg.momentum)
    elif cfg.optimizer == 'SGD':
        net_opt = nn.SGD(net.trainable_params(), learning_rate=cfg.lr)
    else:
        raise ValueError("cfg.optimizer is null")
    net1 = WithLossCell(net, loss)
    net2 = TrainOneStepCell(net1, net_opt)
    model = Model(net2)
    print('Done.')

    print("Train Model...")
    loss_meter = AverageMeter('loss')
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint = ModelCheckpoint(prefix="checkpoint_CTRL", directory=cfg.log_dir, config=config_ck)

    if cfg.mode == context.GRAPH_MODE:
        model.train(cfg.max_epoch, dataset,
                    callbacks=[ckpoint, LossMonitor(), TimeMonitor()], dataset_sink_mode=True)
    else:
        model.train(cfg.max_epoch, dataset,
                    callbacks=[ckpoint, LossMonitor(), TimeMonitor()], dataset_sink_mode=False)
    print('Done.')

    if args.run_cloudbrain:
        batch_size = cfg.test_batch_size
        data = np.random.uniform(0.0, 1.0,
                                 size=[batch_size, cfg.visual_dim + cfg.sentence_embed_dim]).astype(np.float32)
        print(data.shape)
        export(net, Tensor(data), file_name=os.path.join(cfg.log_dir, "TALL"), file_format=args.file_format)
        mox.file.copy_parallel(src_url=_local_train_url, dst_url=args.train_url)
    print("End.")
