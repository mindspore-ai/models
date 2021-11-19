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
"""evaluate imagenet"""

import os
import time

import mindspore.nn as nn
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_dataset
from src.proxylessnas_mobile import proxylessnas_mobile
from src.model_utils.config import config

from src.CrossEntropySmooth import CrossEntropySmooth

if __name__ == '__main__':
    start_time = time.time()

    print('num_classes = ', config.num_classes)

    device_id = config.device_id
    print('device_id = ', device_id)

    if config.use_pynative_mode:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target,
                            device_id=device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                            device_id=device_id, save_graphs=False)

    dataset_path = config.dataset_path
    checkpoint = config.checkpoint
    checkpoint_dir = config.checkpoint_dir

    if config.enable_modelarts:
        # download dataset from obs to cache
        import moxing
        dataset_path = '/cache/dataset'
        if config.data_url.find('/val/') > 0:
            dataset_path += '/val/'
        moxing.file.copy_parallel(src_url=config.data_url, dst_url=dataset_path)

        # download the checkpoint from obs to cache
        if checkpoint != '' and not config.enable_checkpoint_dir:
            base_name = os.path.basename(checkpoint)
            dst_url = '/cache/checkpoint/' + base_name
            moxing.file.copy_parallel(src_url=checkpoint, dst_url=dst_url)
            checkpoint = dst_url

        if checkpoint_dir != '' and config.enable_checkpoint_dir:
            dst_url = '/cache/checkpoint/'
            moxing.file.copy_parallel(src_url=checkpoint_dir, dst_url=dst_url)
            checkpoint_dir = dst_url

    if dataset_path.find('/val') > 0:
        dataset_val_path = dataset_path
    else:
        if config.enable_eval_on_train_dataset:
            dataset_val_path = os.path.join(dataset_path, 'train')
        else:
            dataset_val_path = os.path.join(dataset_path, 'val')

    dataset = create_dataset(dataset_val_path, do_train=False, rank=device_id,
                             group_size=1, batch_size=config.val_batch_size,
                             drop_remainder=config.drop_remainder, shuffle=False)

    if config.enable_label_smooth:
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)
    else:
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net = proxylessnas_mobile(num_classes=config.num_classes)
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss, optimizer=None, metrics=eval_metrics)

    if not config.enable_checkpoint_dir:
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
