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
"""train rednet30."""
import argparse
import os
import time
import mindspore.nn as nn
from mindspore import context
from mindspore import dataset as ds
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from src.dataset import Dataset
from src.model import REDNet30

def train_net(opt):
    """train"""
    device_id = int(os.getenv('DEVICE_ID', '0'))
    rank_id = int(os.getenv('RANK_ID', '0'))
    device_num = int(os.getenv('DEVICE_NUM', '1'))

    if opt.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                            device_target="Ascend", device_id=device_id)

    # if distribute:
    if opt.is_distributed:
        init()
        rank_id = get_rank()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=device_num, gradients_mean=True)

    # dataset
    print("============== Loading Data ==============")
    train_dataset = Dataset(opt.dataset_path, opt.patch_size)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ["input", "label"], num_shards=device_num,
                                           shard_id=rank_id, shuffle=True)
    train_de_dataset = train_de_dataset.batch(opt.batch_size, drop_remainder=True)
    step_size = train_de_dataset.get_dataset_size()

    print("============== Loading Model ==============")
    model = REDNet30()

    optimizer = nn.Adam(model.trainable_params(), learning_rate=opt.lr)
    loss = nn.MSELoss()
    loss_scale_manager = DynamicLossScaleManager(init_loss_scale=opt.init_loss_scale, scale_window=1000)
    model = Model(model, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager, amp_level="O3")
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(keep_checkpoint_max=opt.ckpt_save_max)
    ckpt_cb = ModelCheckpoint(prefix='RedNet30_{}'.format(rank_id),
                              directory=os.path.join("ckpt", 'ckpt_' + str(rank_id) + '/'), config=config_ck)
    cb += [ckpt_cb]

    print("============== Starting Training ==============")
    model.train(opt.num_epochs, train_de_dataset, callbacks=cb, dataset_sink_mode=True)
    print("================== Finished ==================")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/BSD300', help='training image path')
    parser.add_argument('--platform', type=str, default='GPU', choices=('Ascend', 'GPU'), help='run platform')
    parser.add_argument('--is_distributed', type=bool, default=False, help='distributed training')
    parser.add_argument('--patch_size', type=int, default=50, help='training patch size')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--num_epochs', type=int, default=1000, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--ckpt_save_max', type=int, default=5, help='maximum number of checkpoint files can be saved')
    parser.add_argument('--init_loss_scale', type=float, default=65536., help='initialize loss scale')
    option = parser.parse_args()

    set_seed(option.seed)

    time_start = time.time()
    train_net(option)
    time_end = time.time()
    print('train time: %f' % (time_end - time_start))
