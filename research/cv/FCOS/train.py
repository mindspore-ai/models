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
"""FCOS TRAIN"""
import os
import argparse
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.train.callback import TimeMonitor, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore import Model
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size

from src import COCO_dataset
from src.fcos import FCOSDetector
from src.network_define import  WithLossCell, TrainOneStepCell, LossNet
from src.augment import Transforms

set_seed(1)
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("--platform", type=str, default='GPU', help="run platform")
parser.add_argument("--device_num", type=int, default=8, help="device_number to run")
parser.add_argument("--device_id", type=int, default=0, help="DEVICE_ID to run ")
parser.add_argument("--train_path", type=str, default="/data2/dataset/coco2017/train2017")
parser.add_argument("--anno_path", type=str, default="/coco2017/annotations/instances_train2017.json")
parser.add_argument("--ckpt_save_path", type=str, default="/data1/FCOS", help='checkpoint save path')
parser.add_argument("--pretrain_ckpt_path", type=str, default="/data1/test3/src/backbone/s1.ckpt")
opt = parser.parse_args()

def lr_func(_LR_INIT, _WARMUP_STEPS, _WARMUP_FACTOR, _TOTAL_STEPS, _lr_schedule):
    lr_res = []
    for step in range(0, _TOTAL_STEPS):
        _lr = _LR_INIT
        if step < _WARMUP_STEPS:
            alpha = float(step) / _WARMUP_STEPS
            warmup_factor = _WARMUP_FACTOR * (1.0 - alpha) + alpha
            _lr = _lr * warmup_factor
            lr_res.append(_lr)
        else:
            for w in range(len(_lr_schedule)):
                if step < _lr_schedule[w]:
                    lr_res.append(_lr)
                    break
                _lr *= 0.1
            if step >= 160000:
                lr_res.append(_lr)
    return np.array(lr_res, dtype=np.float32)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    context.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")
    dataset_dir = opt.train_path
    annotation_file = opt.anno_path
    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    tr = Transforms()
    if opt.device_num == 1:
        context.set_context(device_id=opt.device_id)
        train_dataset, dataset_size = COCO_dataset.create_coco_dataset(dataset_dir, \
        annotation_file, BATCH_SIZE, shuffle=True, transform=tr)
        rank_id = 0
    else:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, gradients_mean=True, \
        parallel_mode=ParallelMode.DATA_PARALLEL)
        train_dataset, dataset_size = COCO_dataset.create_coco_dataset(dataset_dir, annotation_file, BATCH_SIZE, \
        shuffle=True, transform=tr, num_parallel_workers=device_num, num_shards=device_num, shard_id=rank_id)
    print("the size of the dataset is %d" % train_dataset.get_dataset_size())
    steps_per_epoch = dataset_size//BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMUP_STEPS = 500
    WARMUP_FACTOR = 1.0 / 3.0
    GLOBAL_STEPS = 0
    LR_INIT = 0.01
    lr_schedule = [120000, 160000]
    fcos = FCOSDetector(mode="training", preckpt_path=opt.pretrain_ckpt_path).set_train()
    loss = LossNet()
    lr = Tensor(lr_func(LR_INIT, WARMUP_STEPS, WARMUP_FACTOR, TOTAL_STEPS, lr_schedule))
    sgd_optimizer = nn.SGD(fcos.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=0.0001)
    loss_cb = LossMonitor()
    time_cb = TimeMonitor()
    cb = [loss_cb, time_cb]
    net_with_loss = WithLossCell(fcos, loss)
    net = TrainOneStepCell(net_with_loss, sgd_optimizer)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=10000, keep_checkpoint_max=20)
    save_checkpoint_path = os.path.join(opt.ckpt_save_path, "ckpt_{}/".format(rank_id))
    ckpt_cb = ModelCheckpoint(prefix='ms8p', directory=save_checkpoint_path, config=ckptconfig)
    cb += [ckpt_cb]
    model = Model(net)
    print("successfully build model, and now train the model...")
    model.train(EPOCHS, train_dataset=train_dataset, dataset_sink_mode=True, sink_size=7329, callbacks=cb)
