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
import argparse
import os
import ast
import numpy as np
import mindspore.dataset.vision as C
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
import mindspore.dataset.transforms as C2
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
from original_model import Net

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument('--name', type=str, default="Deepsort", help='Model name')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--device', type=str, default='GPU', choices=("GPU", "Ascend"),
                    help="Device target, support GPU and Ascend.")
parser.add_argument("--num_parallel_workers", help="The number of parallel workers.", type=int, default=8)
parser.add_argument("--pre_train", help='The ckpt file of model.', type=str, default=None)
parser.add_argument("--save_check_point", help="Whether save the training resulting.", type=bool, default=True)
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run distribute')
args = parser.parse_args()

set_seed(1234)
if args.name == "Deepsort":
    from config import config as cfg

def get_lr(base_lr, total_epochs, steps_per_epoch, step_size, gamma):
    lr_each_step = []
    for i in range(1, total_epochs+1):
        if i % step_size == 0:
            base_lr *= gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(base_lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step

target = args.device
if target not in ('GPU', "Ascend"):
    raise ValueError("Unsupported device target.")

context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

device_num = int(os.getenv('RANK_SIZE', '1'))
device_id = int(os.getenv('DEVICE_ID', '0'))
rank = int(os.getenv('RANK_ID', '0'))

if args.run_modelarts:
    import moxing as mox
    cfg.batch_size = cfg.batch_size*int(8/device_num)
    context.set_context(device_id=device_id)
    local_data_url = '/cache/data'
    local_train_url = '/cache/train'
    mox.file.copy_parallel(args.data_url, local_data_url)
    if device_num > 1:
        init()
        context.set_auto_parallel_context(device_num=device_num,\
             parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    DATA_DIR = local_data_url + '/'
elif target == "Ascend":
    if args.run_distribute:
        cfg.batch_size = cfg.batch_size*int(8/device_num)
        context.set_context(device_id=device_id)
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,\
             parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else:
        context.set_context(device_id=device_id)
        device_num = 1
        cfg.batch_size = cfg.batch_size*int(8/device_num)
    DATA_DIR = args.data_url + '/'
elif target == "GPU":
    if args.run_distribute:
        init("nccl")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        context.set_context(device_id=device_id)
    DATA_DIR = args.data_url

data = ds.ImageFolderDataset(DATA_DIR, decode=True, shuffle=True,\
     num_parallel_workers=args.num_parallel_workers, num_shards=device_num, shard_id=rank)

transform_img = [
    C.RandomCrop((128, 64), padding=4),
    C.RandomHorizontalFlip(prob=0.5),
    # Computed from random subset of ImageNet training images
    C.Normalize([0.485*255, 0.456*255, 0.406*255], [0.229*255, 0.224*255, 0.225*255]),
    C.HWC2CHW()
        ]

num_classes = max(data.num_classes(), 0)

type_cast_op = C2.TypeCast(mstype.int32)
data = data.map(input_columns="image", operations=transform_img, num_parallel_workers=args.num_parallel_workers)
data = data.map(input_columns="label", operations=type_cast_op, num_parallel_workers=args.num_parallel_workers)
data = data.batch(batch_size=cfg.batch_size)

data_size = data.get_dataset_size()

loss_cb = LossMonitor(data_size)
time_cb = TimeMonitor(data_size=data_size)
callbacks = [time_cb, loss_cb]

#save training results
if args.save_check_point:
    model_save_path = './ckpt_' + str(rank) + '/'
    config_ck = CheckpointConfig(
        save_checkpoint_steps=data_size, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix='deepsort', directory=model_save_path, config=config_ck)
    callbacks += [ckpoint_cb]

#design learning rate
lr = Tensor(get_lr(cfg.learning_rate, cfg.epoch, data_size, cfg.decay_epoch, cfg.gamma))
# net definition
net = Net(num_classes=num_classes)

# loss and optimizer
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=cfg.momentum)

#train
model = Model(net, loss_fn=loss, optimizer=optimizer)
dataset_sink_mode = True
if target == 'GPU':
    dataset_sink_mode = False
model.train(cfg.epoch, data, callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)
if args.run_modelarts:
    mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
