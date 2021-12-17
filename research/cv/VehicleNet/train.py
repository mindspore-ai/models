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
"""train"""
import ast
import argparse
import os
import time
import mindspore.nn as nn

import mindspore
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.save_callback import SaveCallback
from src.config import common_config, VehicleNet_train, VeRi_train, VeRi_test
from src.dataset import data_to_mindrecord, create_vehiclenet_dataset
from src.VehicleNet_resnet50 import VehicleNet
from src.lr_generator import lr_steps, lr_steps_2

set_seed(1)

def get_base_param(load_ckpt_path):
    """filter parameters"""
    par_dict = load_checkpoint(load_ckpt_path)
    new_params_dict = {}
    for name in par_dict:
        if 'classifier' not in name:
            new_params_dict[name] = par_dict[name]
    return new_params_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VehicleNet train.')
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is false.")
    parser.add_argument('--device_id', type=int, default=None,
                        help='device id of GPU or Ascend. (Default: None)')
    parser.add_argument('--device_num', type=int, default=1, help='Number of device.')
    parser.add_argument('--is_modelarts', type=ast.literal_eval, default=False, help='Train in Modelarts.')
    parser.add_argument('--eval_training', type=ast.literal_eval, default=True, help='Eval in training.')
    parser.add_argument('--ckpt_url', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Location of training outputs.')
    args_opt = parser.parse_args()

    cfg = common_config
    VehicleNet_cfg = VehicleNet_train
    VeRi_cfg = VeRi_train
    VeRi_test_cfg = VeRi_test

    device_target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    if args_opt.run_distribute:
        if device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
            init()
        else:
            raise ValueError("Unsupported platform.")
    else:
        if device_target == "Ascend":
            if args_opt.device_id is not None:
                context.set_context(device_id=args_opt.device_id)
            else:
                context.set_context(device_id=cfg.device_id)
        else:
            raise ValueError("Unsupported platform.")

    train_dataset_path = cfg.dataset_path
    pre_trained_file = cfg.pre_trained_file
    if args_opt.is_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=args_opt.data_url,
                               dst_url='/cache/dataset_train/device_' + os.getenv('DEVICE_ID'))
        zip_command = "unzip -o /cache/dataset_train/device_" + os.getenv('DEVICE_ID') \
                      + "/VehicleNet_mindrecord_v3.zip -d /cache/dataset_train/device_" + os.getenv('DEVICE_ID')
        os.system(zip_command)
        train_dataset_path = '/cache/dataset_train/device_' + os.getenv('DEVICE_ID') + '/VehicleNet/'
        pre_trained_file = '/cache/dataset_train/device_' + os.getenv('DEVICE_ID') + '/resnet50.ckpt'

    mindrecord_dir = cfg.mindrecord_dir
    prefix = "first_train_VehicleNet.mindrecord"
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create mindrecord for train.")
        data_to_mindrecord(train_dataset_path, True, True, False, mindrecord_file)
        print("Create mindrecord done, at {}".format(mindrecord_dir))
    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    dataset = create_vehiclenet_dataset(mindrecord_file, batch_size=VehicleNet_cfg.batch_size,
                                        device_num=args_opt.device_num, is_training=True)

    step_per_epoch_first = dataset.get_dataset_size()

    mindrecord_dir = cfg.mindrecord_dir
    prefix = "test_VehicleNet.mindrecord"
    test_mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(test_mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create mindrecord for test.")
        data_to_mindrecord(eval_dataset_path, False, False, True, test_mindrecord_file)
        print("Create mindrecord done, at {}".format(mindrecord_dir))
    while not os.path.exists(test_mindrecord_file + ".db"):
        time.sleep(5)

    prefix = "query_VehicleNet.mindrecord"
    query_mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(query_mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create mindrecord for query.")
        data_to_mindrecord(eval_dataset_path, False, False, False, query_mindrecord_file)
        print("Create mindrecord done, at {}".format(mindrecord_dir))
    while not os.path.exists(query_mindrecord_file + ".db"):
        time.sleep(5)

    test_dataset = create_vehiclenet_dataset(test_mindrecord_file, batch_size=1, device_num=1, is_training=False)
    query_dataset = create_vehiclenet_dataset(query_mindrecord_file, batch_size=1, device_num=1, is_training=False)
    test_data_num = test_dataset.get_dataset_size()
    query_data_num = query_dataset.get_dataset_size()

    net = VehicleNet(class_num=VehicleNet_cfg.num_classes)

    net_test = VehicleNet(class_num=VeRi_test_cfg.num_classes)
    net_test.classifier.classifier = nn.SequentialCell()

    if cfg.pre_trained:
        param_dict = load_checkpoint(pre_trained_file)
        load_param_into_net(net, param_dict)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    lr_base = lr_steps(0.1 * VehicleNet_cfg.lr_init, VehicleNet_cfg.epoch_size, step_per_epoch_first)
    lr_ignored = lr_steps(VehicleNet_cfg.lr_init, VehicleNet_cfg.epoch_size, step_per_epoch_first)

    base_params = list(filter(lambda x: 'classifier' not in x.name, net.trainable_params()))
    ignored_params = list(filter(lambda x: 'classifier' in x.name, net.trainable_params()))
    train_params = [{'params': base_params, 'lr': Tensor(lr_base, mindspore.float32)},
                    {'params': ignored_params, 'lr': Tensor(lr_ignored, mindspore.float32)}]
    opt = nn.SGD(train_params, momentum=VehicleNet_cfg.momentum, weight_decay=VehicleNet_cfg.weight_decay)

    model = Model(net, loss_fn=loss, optimizer=opt)

    time_cb = TimeMonitor(data_size=step_per_epoch_first)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if cfg.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_epochs * step_per_epoch_first,
                                     keep_checkpoint_max=cfg.keep_checkpoint_max)
        if args_opt.is_modelarts:
            save_checkpoint_path = '/cache/train_output/checkpoint'
            if args_opt.device_num == 1:
                ckpt_cb = ModelCheckpoint(prefix='first_train_vehiclenet',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
            if args_opt.device_num > 1 and get_rank() % 8 == 0:
                if args_opt.eval_training:
                    save_cb = SaveCallback(net_test, test_dataset, query_dataset, 10, VeRi_test_cfg)
                    cb.append(save_cb)
                ckpt_cb = ModelCheckpoint(prefix='first_train_vehiclenet',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
        else:
            save_checkpoint_path = cfg.checkpoint_dir
            if not os.path.isdir(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)

            if args_opt.device_num == 1:
                if args_opt.eval_training:
                    save_cb = SaveCallback(net_test, test_dataset, query_dataset, 10, VeRi_test_cfg)
                    cb.append(save_cb)
                ckpt_cb = ModelCheckpoint(prefix='first_train_vehiclenet',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
            if args_opt.device_num > 1 and get_rank() % 8 == 0:
                if args_opt.eval_training:
                    save_cb = SaveCallback(net_test, test_dataset, query_dataset, 10, VeRi_test_cfg)
                    cb.append(save_cb)
                ckpt_cb = ModelCheckpoint(prefix='first_train_vehiclenet',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]

    model.train(VehicleNet_cfg.epoch_size, dataset, callbacks=cb)
    time.sleep(120)

    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args_opt.train_url)

    mindrecord_dir = cfg.mindrecord_dir
    prefix = "second_train_VehicleNet.mindrecord"
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create mindrecord for train.")
        data_to_mindrecord(train_dataset_path, True, False, False, mindrecord_file)
        print("Create mindrecord done, at {}".format(mindrecord_dir))
    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    dataset = create_vehiclenet_dataset(mindrecord_file, batch_size=VeRi_cfg.batch_size,
                                        device_num=args_opt.device_num, is_training=True)
    step_per_epoch_second = dataset.get_dataset_size()

    net = VehicleNet(class_num=VeRi_cfg.num_classes)


    first_trained_file = '/cache/train_output/checkpoint/first_train_vehiclenet-80_' + \
                         str(step_per_epoch_first) + '.ckpt'
    # first_trained_file = '../../checkpoint/first_train_vehiclenet-80_' + str(step_per_epoch_first) + '.ckpt'

    param_dict = get_base_param(first_trained_file)
    load_param_into_net(net, param_dict)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    lr_base = lr_steps_2(0.1 * VeRi_cfg.lr_init, VeRi_cfg.epoch_size, step_per_epoch_second)
    lr_ignored = lr_steps_2(VeRi_cfg.lr_init, VeRi_cfg.epoch_size, step_per_epoch_second)

    base_params = list(filter(lambda x: 'classifier' not in x.name, net.trainable_params()))
    ignored_params = list(filter(lambda x: 'classifier' in x.name, net.trainable_params()))
    train_params = [{'params': base_params, 'lr': Tensor(lr_base, mindspore.float32)},
                    {'params': ignored_params, 'lr': Tensor(lr_ignored, mindspore.float32)}]

    opt = nn.SGD(train_params, momentum=VeRi_cfg.momentum, weight_decay=VeRi_cfg.weight_decay)

    model = Model(net, loss_fn=loss, optimizer=opt)

    time_cb = TimeMonitor(data_size=step_per_epoch_second)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if cfg.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_epochs * step_per_epoch_second,
                                     keep_checkpoint_max=cfg.keep_checkpoint_max)
        if args_opt.is_modelarts:
            save_checkpoint_path = '/cache/train_output/checkpoint'
            if args_opt.device_num == 1:
                ckpt_cb = ModelCheckpoint(prefix='second_train_vehiclenet',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
            if args_opt.device_num > 1 and get_rank() % 8 == 0:
                if args_opt.eval_training:
                    save_cb = SaveCallback(net_test, test_dataset, query_dataset, 10, VeRi_test_cfg)
                    cb.append(save_cb)
                ckpt_cb = ModelCheckpoint(prefix='second_train_vehiclenet',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
        else:
            save_checkpoint_path = cfg.checkpoint_dir
            if not os.path.isdir(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)

            if args_opt.device_num == 1:
                if args_opt.eval_training:
                    save_cb = SaveCallback(net_test, test_dataset, query_dataset, 5, VeRi_test_cfg)
                    cb.append(save_cb)
                ckpt_cb = ModelCheckpoint(prefix='second_train_vehiclenet',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
            if args_opt.device_num > 1 and get_rank() % 8 == 0:
                if args_opt.eval_training:
                    save_cb = SaveCallback(net_test, test_dataset, query_dataset, 10, VeRi_test_cfg)
                    cb.append(save_cb)
                ckpt_cb = ModelCheckpoint(prefix='second_train_vehiclenet',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]

    model.train(VeRi_cfg.epoch_size, dataset, callbacks=cb)

    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args_opt.train_url)
