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
"""train posenet"""
import ast
import argparse
import os
import shutil
import numpy as np
from mindspore.common import set_seed
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.nn import Adagrad
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
import mindspore.common.dtype as ms
from src.posenet import PoseNet
from src.config import common_config, KingsCollege, StMarysChurch
from src.dataset import create_posenet_dataset
from src.loss import PosenetWithLoss

set_seed(1)

parser = argparse.ArgumentParser(description='Posenet train.')
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute, default is false.")
parser.add_argument('--dataset', type=str, default='KingsCollege',
                    choices=['KingsCollege', 'StMarysChurch'], help='Name of dataset.')
parser.add_argument('--device_num', type=int, default=1, help='Number of device.')
# 模型输出目录
parser.add_argument('--train_url', type=str, default='', help='the path model saved')
# 数据集目录
parser.add_argument('--data_url', type=str, default='', help='the training data')
# 抽取出来的超参配置
parser.add_argument('--pre_trained', type=ast.literal_eval, default=False, help='Pretrained checkpoint path')
parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--max_steps', type=int, default=30000, help='max_steps')
parser.add_argument('--save_checkpoint_epochs', type=int, default=5, help='save_checkpoint_epochs')
parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='keep_checkpoint_max')
parser.add_argument('--save_checkpoint', type=ast.literal_eval, default=True, help='save_checkpoint')
parser.add_argument("--file_name", type=str, default="posenet", help="output file name.")
parser.add_argument('--is_modelarts', type=ast.literal_eval, default=True, help='Train in Modelarts.')
args_opt = parser.parse_args()
CACHE_TRAINING_URL = "/cache/training/"
CACHE = "/cache/"
src = "/"
local_data_path = '/cache/data/'
if not os.path.isdir(CACHE_TRAINING_URL):
    os.makedirs(CACHE_TRAINING_URL)

if __name__ == '__main__':
    cfg = common_config
    if args_opt.dataset == "KingsCollege":
        dataset_cfg = KingsCollege
    elif args_opt.dataset == "StMarysChurch":
        dataset_cfg = StMarysChurch

    device_target = args_opt.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    if args_opt.run_distribute:
        if device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
            init()
        elif device_target == "GPU":
            init()
            context.set_auto_parallel_context(device_num=args_opt.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
    else:
        if args_opt.device_id is not None:
            context.set_context(device_id=args_opt.device_id)
        else:
            context.set_context(device_id=cfg.device_id)

    train_dataset_path = dataset_cfg.dataset_path
    if args_opt.is_modelarts:
        import moxing as mox
        mox.file.copy_parallel(args_opt.data_url, CACHE)
    if args_opt.dataset == "KingsCollege":
        mindrecord_file_name = "KingsCollege_posenet_train.mindrecord"
    elif args_opt.dataset == "StMarysChurch":
        mindrecord_file_name = "StMarysChurch_posenet_train.mindrecord"
    mindrecord_file = os.path.join(CACHE, mindrecord_file_name)
    dataset = create_posenet_dataset(mindrecord_file, batch_size=dataset_cfg.batch_size,
                                     device_num=args_opt.device_num, is_training=True)
    step_per_epoch = dataset.get_dataset_size()

    net_with_loss = PosenetWithLoss(args_opt.pre_trained)
    opt = Adagrad(params=net_with_loss.trainable_params(),
                  learning_rate=dataset_cfg.lr_init,
                  weight_decay=dataset_cfg.weight_decay)
    model = Model(net_with_loss, optimizer=opt)

    time_cb = TimeMonitor(data_size=step_per_epoch)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if args_opt.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoint_epochs * step_per_epoch,
                                     keep_checkpoint_max=args_opt.keep_checkpoint_max)
        if args_opt.is_modelarts:
            save_checkpoint_path = CACHE_TRAINING_URL
            if args_opt.device_num == 1:
                ckpt_cb = ModelCheckpoint(prefix='train_posenet_' + args_opt.dataset,
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
            if args_opt.device_num > 1 and get_rank() % 8 == 0:
                ckpt_cb = ModelCheckpoint(prefix='train_posenet_' + args_opt.dataset,
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
        else:
            save_checkpoint_path = cfg.checkpoint_dir
            if not os.path.isdir(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)

            if args_opt.device_num == 1:
                ckpt_cb = ModelCheckpoint(prefix='train_posenet_' + args_opt.dataset,
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
            if args_opt.device_num > 1 and get_rank() % 8 == 0:
                ckpt_cb = ModelCheckpoint(prefix='train_posenet_' + args_opt.dataset,
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]

    epoch_size = args_opt.max_steps // args_opt.device_num // step_per_epoch
    model.train(1, dataset, callbacks=cb)

    net = PoseNet()
    file_name1 = "train_posenet_KingsCollege-1_16.ckpt"
    assert cfg.checkpoint_dir is not None, "cfg.checkpoint_dir is None."
    param_dict = load_checkpoint(os.path.join(CACHE_TRAINING_URL, file_name1))
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.ones([1, 3, 224, 224]), ms.float32)
    export(net, input_arr, file_name=args_opt.file_name, file_format='AIR')
    shutil.copy('posenet.air', CACHE_TRAINING_URL)
    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url=CACHE_TRAINING_URL, dst_url=args_opt.train_url)
