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
The Module of Train
"""
import os.path
import argparse
import yaml
from yaml import Loader

from mindspore.train.model import Model
import mindspore.context as context
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor

from src.model.SRFlow import SRFlowNetNllFor
from src.dataloader import create_train_dataset
from src.scheduler.scheduler import step_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YMAL file.')
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    context.set_context(device_target="GPU")
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(max_call_depth=3000)

    dataset_opt = opt['datasets']['train']
    train_opt = opt['train']

    dataset = create_train_dataset(opt=opt, group_size=1, rank_id=0)

    step = dataset.get_dataset_size()

    total_epoch = dataset_opt['epoch']
    weight_decay = train_opt['weight_decay_G']
    beta1 = train_opt['beta1']
    beta2 = train_opt['beta2']
    lr_setps_val = opt['train']['lr_steps_rel']
    lr_decay = opt['train']['lr_decay']
    lr_init = opt['train']['lr_init']

    total_steps = total_epoch * step
    milestone = []
    for i in lr_setps_val:
        milestone.append(int(total_steps * i))
    loss_net = SRFlowNetNllFor(opt=opt)

    if os.path.exists(opt['path']['train_pretrain_model_G']):
        ckpt_file_name = opt['path']['train_pretrain_model_G']
        param_dict = load_checkpoint(ckpt_file_name)
        load_param_into_net(loss_net, param_dict)

    optim_params_RRDB = []
    optim_params_other = []
    for param in loss_net.trainable_params():
        if param.requires_grad:
            if '.RRDB.' in param.name:
                optim_params_RRDB.append(param)
            else:
                optim_params_other.append(param)

    scheduler = step_lr(lr_init, milestone, lr_decay, total_steps)

    group_params = (
        [
            {"params": optim_params_other, "lr": train_opt['lr_G'], "weight_decay": weight_decay},
            {"params": optim_params_RRDB, "lr": train_opt['lr_G'], "weight_decay": weight_decay}
        ]
    )

    optimizer = nn.Adam(group_params, learning_rate=scheduler, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                        loss_scale=128.0)

    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    train_net.set_train()

    config_ck = CheckpointConfig(save_checkpoint_steps=10000, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix='SRFlow', directory='Pretrainedmodel/ckpt', config=config_ck)
    callbacks = [TimeMonitor(data_size=step), LossMonitor(), ckpoint_cb]

    model = Model(loss_net, loss_scale_manager=None, optimizer=optimizer)
    model.train(total_epoch, dataset, callbacks=callbacks, dataset_sink_mode=False)


if __name__ == '__main__':
    main()
