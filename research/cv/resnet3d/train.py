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
"""train for resnet3d."""
import os
import random
from pathlib import Path
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore import dataset as de
from mindspore.common import set_seed
import mindspore.common.initializer as weight_init
from mindspore.communication.management import init
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import (ModelCheckpoint, CheckpointConfig,
                                      LossMonitor, TimeMonitor)
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.lr import get_lr
from src.ResNet3D import generate_model
from src.save_callback import SaveCallback
from src.loss import CrossEntropySmooth
from src.dataset import create_train_dataset, create_eval_dataset
from src.config import config as args_opt
# from src.config import config_ucf101, config_hmdb51

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)
set_seed(1)

if __name__ == '__main__':
    target = args_opt.device_target
    cfg = args_opt
    if cfg.n_classes == 101:
        dataset_name = "ucf101"
    elif cfg.n_classes == 51:
        dataset_name = 'hmdb51'
    else:
        dataset_name = ""
    print("==============================config======================")
    print(cfg)
    print("==============================config======================")
    if args_opt.is_modelarts:
        import moxing as mox
    # init context
    if args_opt.mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=target, save_graphs=False)
    else:
        context.set_context(mode=context.PYNATIVE_MODE,
                            device_target=target, save_graphs=False)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id,
                                enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(
                parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            init()
    else:
        if target == "Ascend":
            device_id = args_opt.device_id
            context.set_context(device_id=device_id)

    # create dataset
    if args_opt.is_modelarts:
        root_path = '/cache/data_' + os.getenv('DEVICE_ID')
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=root_path)
        unzip_command = "unzip -o -q " + root_path + '/jpg.zip' \
                                                     " -d " + root_path + '/jpg'
        os.system(unzip_command)
        cfg.video_path = Path(root_path + "/jpg")
        if dataset_name == 'ucf101':
            cfg.annotation_path = Path(root_path + '/json/ucf101_01.json')
        else:
            cfg.annotation_path = Path(root_path + '/json/hmdb51_1.json')
        cfg.pretrain_path = root_path + "/pretrain.ckpt"
        cfg.result_path = root_path + '/train_output/'

    train_dataset = create_train_dataset(cfg.video_path, cfg.annotation_path, cfg,
                                         batch_size=cfg.batch_size, target="Ascend")
    if cfg.eval_in_training:
        inference_dataset = create_eval_dataset(
            cfg.video_path, cfg.annotation_path, cfg)

    step_size = train_dataset.get_dataset_size()

    # load pre_trained model and define net

    net = generate_model(n_classes=cfg.n_classes, stop_weights_update=True,
                         sync=args_opt.run_distribute)   # args_opt.run_distribute
    param_dict = load_checkpoint(cfg.pretrain_path)
    load_param_into_net(net, param_dict)

    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))

    # init lr
    lr = get_lr(lr_init=cfg.lr_init, lr_end=cfg.lr_end, lr_max=cfg.lr_max,
                warmup_epochs=cfg.warmup_epochs, total_epochs=cfg.n_epochs,
                steps_per_epoch=step_size, lr_decay_mode=cfg.lr_decay_mode)

    # define opt
    param_to_train = []
    if cfg.start_ft == 'conv1':
        param_to_train = net.trainable_params()
    else:
        start = False
        for param in net.trainable_params():
            if not start and cfg.start_ft in param.name:
                start = True
            if start:
                param_to_train.append(param)

    net_opt = nn.SGD(param_to_train, lr, momentum=cfg.momentum,
                     weight_decay=cfg.weight_decay, nesterov=True)

    loss = CrossEntropySmooth(smooth_factor=0.1, num_classes=cfg.n_classes)
    model = Model(net, loss_fn=loss, optimizer=net_opt)

    # define callbacks
    cb = []
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_epochs * step_size,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(
        prefix="resnet3d-" + dataset_name, directory=cfg.result_path, config=config_ck)

    if args_opt.run_distribute:
        if cfg.eval_in_training and os.getenv('DEVICE_ID') == 0:
            save_cb = SaveCallback(model, inference_dataset, 20, cfg)
            cb.append(save_cb)
    else:
        if cfg.eval_in_training:
            save_cb = SaveCallback(model, inference_dataset, 20, cfg)
            cb.append(save_cb)

    cb += [time_cb, loss_cb, ckpt_cb]

    print("=======Training Begin========")
    model.train(cfg.n_epochs, train_dataset,
                callbacks=cb, dataset_sink_mode=True)

    if args_opt.is_modelarts:
        mox.file.copy_parallel(src_url=cfg.result_path,
                               dst_url=args_opt.train_url)
