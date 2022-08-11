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

import os
import glob
import argparse
import numpy as np

import moxing as mox
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as C

from mindspore.common import set_seed
from mindspore.common import dtype as mstype
from mindspore.communication.management import init
from mindspore.train.model import Model, ParallelMode
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor

from src.deep.original_model import Net
set_seed(1234)

def export_deepsort(checkpoint_path, s_prefix, file_name, file_format):
    """ export_stgcn """
    # load checkpoint
    net_export = Net(reid=True, ascend=True)
    prob_ckpt_list = os.path.join(checkpoint_path, "{}*.ckpt".format(s_prefix))
    ckpt_list = glob.glob(prob_ckpt_list)
    if not ckpt_list:
        print('Freezing model failed!')
        print("can not find ckpt files. ")
    else:
        ckpt_list.sort(key=os.path.getmtime)
        ckpt_name = ckpt_list[-1]
        print("checkpoint file name", ckpt_name)
        param_dict = load_checkpoint(ckpt_name)
        load_param_into_net(net_export, param_dict)

        input_x = Tensor(np.zeros([1, 3, 128, 64]), mstype.float32)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        file_name = os.path.join(checkpoint_path, file_name)
        export(net_export, input_x, file_name=file_name, file_format=file_format)
        print('Freezing model success!')
    return 0

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train on market1501")
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument("--epoch", help="Path to custom detections.", type=int, default=100)
    parser.add_argument("--batch_size", help="Batch size for Training.", type=int, default=8)
    parser.add_argument("--num_parallel_workers", help="The number of parallel workers.", type=int, default=16)
    parser.add_argument("--save_check_point", help="Whether save the training resulting.", type=bool, default=True)

    #learning rate
    parser.add_argument("--learning_rate", help="Learning rate.", type=float, default=0.1)
    parser.add_argument("--decay_epoch", help="decay epochs.", type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.10, help='learning rate decay.')
    parser.add_argument("--momentum", help="", type=float, default=0.9)

    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend. (Default: 0)')
    #export network
    parser.add_argument("--file_name", type=str, default="deepsort", help="output file name.")
    parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")

    return parser.parse_args()
def get_lr(base_lr, total_epochs, steps_per_epoch, step_size, gamma):
    lr_each_step = []
    for i in range(1, total_epochs+1):
        if i % step_size == 0:
            base_lr *= gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(base_lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


args = parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)


device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
args.batch_size = args.batch_size*int(8/device_num)
context.set_context(device_id=device_id)
local_data_url = '/cache/data'
local_train_url = '/cache/train'
mox.file.copy_parallel(args.data_url, local_data_url)
if device_num > 1:
    init()
    context.set_auto_parallel_context(device_num=device_num,\
            parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
DATA_DIR = local_data_url + '/'

data = ds.ImageFolderDataset(DATA_DIR, decode=True, shuffle=True,\
     num_parallel_workers=args.num_parallel_workers, num_shards=device_num, shard_id=device_id)

transform_img = [
    C.RandomCrop((128, 64), padding=4),
    C.RandomHorizontalFlip(prob=0.5),
    # Computed from random subset of ImageNet training images
    C.Normalize([0.485*255, 0.456*255, 0.406*255], [0.229*255, 0.224*255, 0.225*255]),
    C.HWC2CHW()
        ]

num_classes = max(data.num_classes(), 0)

data = data.map(input_columns="image", operations=transform_img, num_parallel_workers=args.num_parallel_workers)
data = data.batch(batch_size=args.batch_size)

data_size = data.get_dataset_size()

loss_cb = LossMonitor(data_size)
time_cb = TimeMonitor(data_size=data_size)
callbacks = [time_cb, loss_cb]

#save training results
prefix = "deepsort"
if args.save_check_point and (device_num == 1 or device_id == 0):

    config_ck = CheckpointConfig(
        save_checkpoint_steps=data_size*args.epoch, keep_checkpoint_max=args.epoch)

    ckpoint_cb = ModelCheckpoint(prefix=prefix, directory=local_train_url, config=config_ck)

    callbacks += [ckpoint_cb]

#design learning rate
lr = Tensor(get_lr(args.learning_rate, args.epoch, data_size, args.decay_epoch, args.gamma))
# net definition
net = Net(num_classes=num_classes)

# loss and optimizer

loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=args.momentum)

#train
model = Model(net, loss_fn=loss, optimizer=optimizer)
model.train(args.epoch, data, callbacks=callbacks, dataset_sink_mode=True)

#export deepsort
export_deepsort(local_train_url, prefix, args.file_name, args.file_format)

mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
