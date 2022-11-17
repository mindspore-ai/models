# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""evaluate_imagenet"""
import argparse
import ast
import os
import time

import mindspore.nn as nn

from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config_gpu, config_ascend, config_cpu
from src.dataset import create_dataset
from src.shufflenetv2 import ShuffleNetV2
from src.CrossEntropySmooth import CrossEntropySmooth

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='image classification evaluation')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='the checkpoint of ShuffleNetV2 (Default: None)')
    parser.add_argument('--enable_checkpoint_dir', type=ast.literal_eval, default=False,
                        help='whether to evaluate the checkpoints in checkpoint_dir(Default:False)')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='the directory that contains one or more checkpoint files(Default: None)')
    parser.add_argument('--dataset_path', type=str, default='../imagenet', help='Dataset path')
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'),
                        help='run platform(Default:Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='device id(Default:0)')

    parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False)
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path for modelarts')
    parser.add_argument('--train_url', type=str, default=None, help='Output path for modelarts')

    parser.add_argument('--use_pynative_mode', type=ast.literal_eval, default=True,
                        help='whether to use pynative mode for device(Default: False)')
    parser.add_argument('--normalize', type=ast.literal_eval, default=True,
                        help='whether to normalize the dataset(Default: True)')

    parser.add_argument('--enable_tobgr', type=ast.literal_eval, default=False,
                        help='whether to use the toBGR()(Default: False)')

    parser.add_argument('--use_nn_default_loss', type=ast.literal_eval, default=True,
                        help='whether to use nn.SoftmaxCrossEntropyWithLogits(Default: True)')

    parser.add_argument('--overwrite_config', type=ast.literal_eval, default=False,
                        help='whether to overwrite the config according to the arguments')
    #when the overwrite_config == True , the following argument will be written to config
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--val_batch_size', type=int, default=125, help='batch size for evaluation')

    args_opt = parser.parse_args()

    if args_opt.platform == 'GPU':
        config = config_gpu
        drop_remainder = True
    elif args_opt.platform == 'CPU':
        config = config_cpu
        drop_remainder = True
    else:
        config = config_ascend
        drop_remainder = False

    if args_opt.overwrite_config:
        config.num_classes = args_opt.num_classes
        config.val_batch_size = args_opt.val_batch_size

    print('num_classes = ', config.num_classes)
    print('val_batch_size = ', config.val_batch_size)

    device_id = args_opt.device_id
    print('device_id = ', device_id)

    if args_opt.use_pynative_mode:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.platform,
                            device_id=device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.platform,
                            device_id=device_id, save_graphs=False)

    dataset_path = args_opt.dataset_path
    checkpoint = args_opt.checkpoint
    checkpoint_dir = args_opt.checkpoint_dir

    if args_opt.is_modelarts:
        # download dataset from obs to cache
        import moxing
        dataset_path = '/cache/dataset'
        if args_opt.data_url.find('/val/') > 0:
            dataset_path += '/val/'

        print('data_url = ', args_opt.data_url)
        print('dataset_path = ', dataset_path)

        moxing.file.copy_parallel(src_url=args_opt.data_url, dst_url=dataset_path)

        # download the checkpoint from obs to cache
        if checkpoint != '' and not args_opt.enable_checkpoint_dir:
            base_name = os.path.basename(checkpoint)
            dst_url = '/cache/checkpoint/' + base_name
            moxing.file.copy_parallel(src_url=checkpoint, dst_url=dst_url)
            checkpoint = dst_url

        if checkpoint_dir != '' and args_opt.enable_checkpoint_dir:
            dst_url = '/cache/checkpoint/'
            moxing.file.copy_parallel(src_url=checkpoint_dir, dst_url=dst_url)
            checkpoint_dir = dst_url

    if dataset_path.find('/val') > 0:
        dataset_val_path = dataset_path
    else:
        dataset_val_path = os.path.join(dataset_path, 'val')
        if not os.path.exists(dataset_val_path):
            dataset_val_path = dataset_path

    print('dataset_val_path = ', dataset_val_path)

    if args_opt.platform != 'CPU':
        dataset = create_dataset(dataset_val_path, do_train=False, rank=device_id,
                                 group_size=1, batch_size=config.val_batch_size,
                                 drop_remainder=drop_remainder, shuffle=False,
                                 normalize=args_opt.normalize,
                                 enable_tobgr=args_opt.enable_tobgr)
    else:
        dataset = create_dataset(dataset_val_path, do_train=False, rank=device_id,
                                 group_size=1, batch_size=config.val_batch_size,
                                 drop_remainder=drop_remainder, shuffle=False,
                                 normalize=args_opt.normalize,
                                 enable_tobgr=args_opt.enable_tobgr,
                                 num_parallel_workers=config.num_parallel_workers)
    if not args_opt.use_nn_default_loss:
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net = ShuffleNetV2(n_class=config.num_classes)

    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss, optimizer=None, metrics=eval_metrics)

    if not args_opt.enable_checkpoint_dir:
        print(checkpoint)

        ckpt = load_checkpoint(checkpoint)
        load_param_into_net(net, ckpt)
        net.set_train(False)

        metrics = model.eval(dataset)

        print("metric: ", metrics)

        print("time: ", (time.time() - start_time) / 60, " minutes")
    else:
        file_list = []
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.ckpt':
                    file_list.append(os.path.join(root, file))

        file_count = 0

        best_top1_acc = 0.0
        best_top1_acc_checkpoint = ''

        file_list.sort()

        for checkpoint in file_list:
            if not os.path.exists(checkpoint):
                continue

            file_count += 1
            print(checkpoint)

            ckpt = load_checkpoint(checkpoint)
            load_param_into_net(net, ckpt)
            net.set_train(False)

            metrics = model.eval(dataset)

            print("metric: ", metrics)
            top1_acc = metrics['Top1-Acc']

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top1_acc_checkpoint = checkpoint
                print('*********************************************************************')

        print('*********************************************************************')
        print(file_count, ' checkpoints have been evaluated')
        print('Best Top1-Acc is ', best_top1_acc, ' on ', best_top1_acc_checkpoint)
        print('time: ', (time.time() - start_time) / 60, ' minutes')
        print('*********************************************************************')
