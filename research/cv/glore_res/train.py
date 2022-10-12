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
#################train glore_resnet series on Imagenet2012########################
python train.py
"""

import os
import random
import numpy as np

import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from mindspore import Tensor
from mindspore import context
from mindspore import dataset as de
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.metrics import Accuracy
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.common import set_seed

from src.lr_generator import get_lr
from src.config import config
from src.glore_resnet import glore_resnet200, glore_resnet50, glore_resnet101
from src.dataset import create_dataset_ImageNet as get_dataset
from src.dataset import create_train_dataset, create_eval_dataset, _get_rank_info
from src.loss import SoftmaxCrossEntropyExpand, CrossEntropySmooth
from src.autoaugment import autoaugment
from src.save_callback import SaveCallback

if config.isModelArts:
    import moxing as mox
if config.net == 'resnet200' or config.net == 'resnet101':
    if config.device_target == "GPU":
        config.cast_fp16 = False

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)
if config.net == 'resnet200' or config.net == 'resnet101' or config.net == 'resnet50':
    set_seed(1)

if __name__ == '__main__':

    target = config.device_target
    ckpt_save_dir = config.save_checkpoint_path
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if config.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
            if config.net == 'resnet200':
                context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True,
                                                  auto_parallel_search_mode="recursive_programming")
            elif config.net == 'resnet50':
                context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
            init()
        elif target == "GPU":
            init()
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        device_id = config.device_id
        context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False,
                            device_id=device_id)
    # get device_num, device_id after device init
    device_num, device_id = _get_rank_info()
    #create dataset
    train_dataset_path = os.path.abspath(config.data_url)
    eval_dataset_path = os.path.abspath(config.eval_data_url)

    # download dataset from obs to cache if train on ModelArts
    if config.net == 'resnet50':
        if config.isModelArts:
            mox.file.copy_parallel(src_url=config.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
            train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/train'
            eval_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/val'
        if config.use_autoaugment:
            print("===========Use autoaugment==========")
            train_dataset = autoaugment(dataset_path=train_dataset_path, repeat_num=1,
                                        batch_size=config.batch_size, target=target)
        else:
            train_dataset = create_train_dataset(dataset_path=train_dataset_path, repeat_num=1,
                                                 batch_size=config.batch_size, target=target)
        eval_dataset = create_eval_dataset(dataset_path=eval_dataset_path, repeat_num=1, batch_size=config.batch_size)
    elif config.net == 'resnet200' or config.net == 'resnet101':
        if config.isModelArts:
            # download dataset from obs to cache
            mox.file.copy_parallel(src_url=config.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
            train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID')
        # create dataset
        else:
            train_dataset = get_dataset(dataset_path=train_dataset_path, do_train=True, use_randaugment=True,
                                        repeat_num=1, batch_size=config.batch_size, target=target)


    step_size = train_dataset.get_dataset_size()
    #define net
    if config.net == 'resnet50':
        net = glore_resnet50(class_num=config.class_num, use_glore=config.use_glore)
    elif config.net == 'resnet200':
        net = glore_resnet200(cast_fp16=config.cast_fp16, class_num=config.class_num, use_glore=config.use_glore)
    elif config.net == 'resnet101':
        net = glore_resnet101(cast_fp16=config.cast_fp16, class_num=config.class_num, use_glore=config.use_glore)
    # init weight
    if config.pretrained_ckpt:
        param_dict = load_checkpoint(config.pretrained_ckpt)
        load_param_into_net(net, param_dict)
    if config.net == 'resnet50':
        for _, cell in net.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Conv1d)):
                if config.weight_init == 'xavier_uniform':
                    cell.weight.default_input = weight_init.initializer(weight_init.XavierUniform(),
                                                                        cell.weight.shape,
                                                                        cell.weight.dtype)
                elif config.weight_init == 'he_uniform':
                    cell.weight.default_input = weight_init.initializer(weight_init.HeUniform(),
                                                                        cell.weight.shape,
                                                                        cell.weight.dtype)
                else:  # config.weight_init == 'he_normal' or the others
                    cell.weight.default_input = weight_init.initializer(weight_init.HeNormal(),
                                                                        cell.weight.shape,
                                                                        cell.weight.dtype)

            if isinstance(cell, nn.Dense):
                cell.weight.default_input = weight_init.initializer(weight_init.TruncatedNormal(),
                                                                    cell.weight.shape,
                                                                    cell.weight.dtype)
    elif config.net == 'resnet200' or config.net == 'resnet101':
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.default_input = weight_init.initializer(weight_init.XavierUniform(),
                                                                    cell.weight.shape,
                                                                    cell.weight.dtype)
            if isinstance(cell, nn.Dense):
                cell.weight.default_input = weight_init.initializer(weight_init.TruncatedNormal(),
                                                                    cell.weight.shape,
                                                                    cell.weight.dtype)
    # init lr
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)
    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    if config.net == 'resnet50':
        net_opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    elif config.net == 'resnet200' or config.net == 'resnet101':
        net_opt = nn.SGD(group_params, learning_rate=lr, momentum=config.momentum, weight_decay=config.weight_decay,
                         loss_scale=config.loss_scale, nesterov=True)
    # define loss, model
    if config.use_label_smooth:
        loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=config.label_smooth_factor,
                                  num_classes=config.class_num)
    else:
        loss = SoftmaxCrossEntropyExpand(sparse=True)
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    if config.device_target == 'Ascend':
        model = Model(net, loss_fn=loss, optimizer=net_opt, loss_scale_manager=loss_scale,
                      metrics={"Accuracy": Accuracy()})
    elif config.device_target == 'GPU':
        if config.net == 'resnet50':
            model = Model(net, loss_fn=loss, optimizer=net_opt, loss_scale_manager=loss_scale,
                          amp_level="O2", metrics={"Accuracy": Accuracy()})
        elif config.net == 'resnet200' or config.net == 'resnet101':
            model = Model(net, loss_fn=loss, optimizer=net_opt, loss_scale_manager=loss_scale)
    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.net == 'resnet50':
        if config.save_checkpoint:
            if config.isModelArts:
                save_checkpoint_path = '/cache/train_output/device_' + os.getenv('DEVICE_ID') + '/'
            else:
                save_checkpoint_path = config.save_checkpoint_path + str(device_id) + '/'
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="glore_resnet50", directory=save_checkpoint_path, config=config_ck)
            save_cb = SaveCallback(model, eval_dataset, save_checkpoint_path)
            cb += [ckpt_cb, save_cb]
    elif config.net == 'resnet200':
        if config.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            if config.isModelArts:
                save_checkpoint_path = '/cache/train_output/device_' + os.getenv('DEVICE_ID') + '/'
                if config.device_num == 1:
                    ckpt_cb = ModelCheckpoint(prefix='glore_resnet200',
                                              directory=save_checkpoint_path,
                                              config=config_ck)
                    cb += [ckpt_cb]
                if config.device_num > 1 and get_rank() % 8 == 0:
                    ckpt_cb = ModelCheckpoint(prefix='glore_resnet200',
                                              directory=save_checkpoint_path,
                                              config=config_ck)
                    cb += [ckpt_cb]
            else:
                save_checkpoint_path = config.save_checkpoint_path + str(device_id) + '/'
                ckpt_cb = ModelCheckpoint(prefix='glore_resnet200',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]

    elif config.net == 'resnet101':
        if config.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            if config.isModelArts:
                save_checkpoint_path = '/cache/train_output/device_' + os.getenv('DEVICE_ID') + '/'
                if config.device_num == 1:
                    ckpt_cb = ModelCheckpoint(prefix='glore_resnet101',
                                              directory=save_checkpoint_path,
                                              config=config_ck)
                    cb += [ckpt_cb]
                if config.device_num > 1 and get_rank() % 8 == 0:
                    ckpt_cb = ModelCheckpoint(prefix='glore_resnet101',
                                              directory=save_checkpoint_path,
                                              config=config_ck)
                    cb += [ckpt_cb]
            else:
                save_checkpoint_path = config.save_checkpoint_path + str(device_id) + '/'
                ckpt_cb = ModelCheckpoint(prefix='glore_resnet101',
                                          directory=save_checkpoint_path,
                                          config=config_ck)
                cb += [ckpt_cb]
    # train model
    print("=======Training Begin========")
    model.train(config.epoch_size - config.pretrain_epoch_size, train_dataset, callbacks=cb, dataset_sink_mode=True)

    # copy train result from cache to obs
    if config.isModelArts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=config.train_url)
