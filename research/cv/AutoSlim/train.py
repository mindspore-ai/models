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
"""
Train the Autoslim resnet
"""
import os
import argparse

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import Model
from mindspore.common import set_seed
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init, get_rank, get_group_size

from src.dataset import data_transforms, create_dataset
from src.lr_generator import get_lr
from src.autoslim_resnet import AutoSlimModel

set_seed(1)

def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='AutoSlim MindSpore Training')
    # Define parameters
    # run on modelarts
    parser.add_argument('--data_url', type=str, default='', help='')
    parser.add_argument('--train_url', type=str, default='', help='')

    # device
    parser.add_argument('--run_modelarts', type=bool, default=False, help='')
    parser.add_argument('--run_distribute', type=bool, default=False, help='choice of distribute train')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='number of device which is chosen')
    parser.add_argument('--device_num', type=int, default=1, help='')

    # train parameters
    parser.add_argument('--dataset_path', type=str, default='/path/to/imagenet', help='The path of your imagenet-1k')
    parser.add_argument('--data_transforms', type=str, default='imagenet1k_mobile', help='')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 256)')
    parser.add_argument('--epoch_size', type=int, default=120, help='number of epoch (default: 100)')
    parser.add_argument('--lr_init', type=float, default=0, help='initialization of learning rate (default: 0)')
    parser.add_argument('--lr_max', type=float, default=0.07, help='maximum of learning rate (default: 0.1)')
    parser.add_argument('--lr_end', type=float, default=0, help='end of learning rate (default: 0)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs for lr (default: 10)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for sgd (default: 1e-4)')

    # save checkpoint
    parser.add_argument('--is_save_checkpoint', type=bool, default=True,
                        help='whether get the checkpoint for evaluation (default: True)')
    parser.add_argument('--save_checkpoint_interval', type=int, default=5005,
                        help='The interval for saving the checkpoints')
    parser.add_argument('--save_checkpoint_max', type=int, default=10, help='The interval for saving the checkpoints')
    parser.add_argument('--save_checkpoint_path', type=str, default='./model', help='The path to save checkpoints')
    parser.add_argument('--test_only', type=bool, default=False, help='only test the dataset with pretained model')
    parser.add_argument('--pretained_checkpoint_path', type=str, default='./AutoSlim-pretrained.ckpt',
                        help='The path of checkpoint for test-only or resume-train')
    args = parser.parse_args()

    return args

def main():
    """train model"""
    # before training, we should set some arguments
    args = preLauch()

    # Define devices
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_train_url = '/cache/ckpt'
        rank_id = 0
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.HYBRID_PARALLEL,
                                              gradients_mean=True)
            rank_id = get_rank()
            local_data_url = os.path.join(local_data_url, str(rank_id))
        mox.file.copy_parallel(args.data_url, local_data_url)
    else:
        if args.run_distribute:
            if args.device_target == 'Ascend':
                device_id = int(os.getenv('DEVICE_ID'))
                device_num = int(os.getenv('RANK_SIZE'))
                context.set_context(device_id=device_id)
                init(backend_name='hccl')
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num, \
                    parallel_mode=ParallelMode.HYBRID_PARALLEL, gradients_mean=True)
                rank_id = get_rank()
            elif args.device_target == 'GPU':
                init('nccl')
                context.reset_auto_parallel_context()
                rank_id = get_rank()
                device_num = get_group_size()
                context.set_auto_parallel_context(device_num=device_num, \
                    parallel_mode=ParallelMode.HYBRID_PARALLEL, gradients_mean=True)
        else:
            device_id = args.device_id
            device_num = 1
            rank_id = 0
            if args.device_target == 'Ascend':
                context.set_context(device_id=device_id)

    # Build network
    net = AutoSlimModel()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_transforms, val_transforms = data_transforms(args)
    if args.run_modelarts:
        train_set, val_set = create_dataset(train_transforms, val_transforms, args, dataset_path=local_data_url,
                                            device_num=device_num, rank_id=rank_id)
    else:
        train_set, val_set = create_dataset(train_transforms, val_transforms, args, dataset_path=args.dataset_path,
                                            device_num=device_num, rank_id=rank_id)
    if not args.test_only:
        step_size = train_set.get_dataset_size()
    else:
        step_size = val_set.get_dataset_size()
    if device_num > 1:
        args.lr_max = args.lr_max * 8
    lr = Tensor(get_lr(lr_init=args.lr_init, lr_end=args.lr_end, lr_max=args.lr_max, warmup_epochs=args.warmup_epochs,
                       total_epochs=args.epoch_size, steps_per_epoch=step_size, lr_decay_mode='cosine'))

    optimizer = nn.SGD(net.trainable_params(), learning_rate=lr, momentum=args.momentum,
                       weight_decay=args.weight_decay, nesterov=True)
    model = Model(net, loss_fn, optimizer, amp_level='O3')

    # Define checkpoint
    loss_cb = LossMonitor(step_size)
    time_cb = TimeMonitor(step_size)
    callbacks = [loss_cb, time_cb]
    if args.is_save_checkpoint and (device_num == 1 or rank_id == 0):
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_interval,
                                       keep_checkpoint_max=args.save_checkpoint_max)
        if args.run_modelarts:
            ckpt_cb = ModelCheckpoint(f"AutoSlim-rank{rank_id}", directory=local_train_url, config=ckpt_config)
        else:
            ckpt_cb = ModelCheckpoint(f"AutoSlim-rank{rank_id}", directory=args.save_checkpoint_path,
                                      config=ckpt_config)
        callbacks += [ckpt_cb]

    print('Initialization may take a long time, please wait patiently...')
    model.train(args.epoch_size, train_set, callbacks)
    print("Finish training.")

    if args.run_modelarts and args.is_save_checkpoint and (device_num == 1 or rank_id == 0):
        mox.file.copy_parallel(local_train_url, args.train_url)

if __name__ == "__main__":
    main()
