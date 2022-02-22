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
"""Lite-HRNet training."""
import os
import ast
import argparse

from mindspore import context, Model, save_checkpoint, load_checkpoint, nn, Tensor
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from src.callback import TimeLossMonitor
from src.model import get_posenet_model, LiteHRNetWithLoss
from src.loss import JointsMSELoss
from src.utils import get_param_groups
from src.lr_scheduler import get_lr
from src.mmpose.topdown_coco_dataset import get_keypoints_coco_dataset
from src.config import experiment_cfg, model_cfg


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description="Mindspore HRNet Training Configurations.")
    parser.add_argument("--train_url", type=str, default='./checkpoints/', help="Storage path of training results.")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Use one card or multiple cards training.")
    parser.add_argument("--device_id", type=int, default=0)

    return parser.parse_args()


def main():
    """Training process."""
    set_seed(1)
    args = parse_args()

    local_train_url = args.train_url
    model_name = experiment_cfg['model_config']
    batch_size = model_cfg.data['samples_per_gpu']

    device_id = args.device_id
    device = experiment_cfg['device']
    if device == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    elif device == 'Ascend':
        raise ValueError('Ascend training not implemented')
    else:
        raise ValueError(f'Unknown device type: {device}. Only "GPU" device type implemented')

    # Print configuration to log
    print('CONFIG_FILE')
    for cfg_item in experiment_cfg.items():
        print(f'{cfg_item[0]} = {cfg_item[1]}')

    for cfg_item in model_cfg.__dir__():
        if cfg_item[0] != '_':
            item = getattr(model_cfg, cfg_item)
            print(f'{cfg_item} = {item}')
    print('CONFIG_FILE_END')

    # Create dataset
    if args.run_distribute:
        init('nccl')
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True)
        loader, _ = get_keypoints_coco_dataset(model_cfg.data['train'],
                                               batch_size,
                                               distributed=True,
                                               random_seed=experiment_cfg['random_seed'])
    else:
        context.set_context(device_id=device_id)
        loader, _ = get_keypoints_coco_dataset(model_cfg.data['train'],
                                               batch_size,
                                               distributed=False,
                                               random_seed=experiment_cfg['random_seed'])

    loss = JointsMSELoss()
    # Create network
    backbone = get_posenet_model(model_cfg.model)
    net = LiteHRNetWithLoss(backbone, loss)
    if experiment_cfg['checkpoint_path'] is not None:
        load_checkpoint(experiment_cfg['checkpoint_path'], net=net)
    net.set_train()

    steps_per_epoch = loader.get_dataset_size()

    # Learning rate adjustment with linear scaling rule
    if args.run_distribute:
        n_gpus = get_group_size()
        lr_default = experiment_cfg['learning_rate']
        print(f'Adjusting learning rate (default = {lr_default}) to {n_gpus} GPUs')
        experiment_cfg['learning_rate'] = experiment_cfg['learning_rate'] * n_gpus

    # Learning rate schedule construction
    lr_args = {'lr': experiment_cfg['learning_rate'], 'lr_epochs': model_cfg.lr_config['step'],
               'steps_per_epoch': steps_per_epoch,
               'warmup_epochs': model_cfg.lr_config['warmup_iters'] / steps_per_epoch,
               'max_epoch': model_cfg.total_epochs, 'lr_gamma': 0.1}
    lr = get_lr(lr_args)
    if experiment_cfg['start_epoch'] > 0:
        lr = lr[int(experiment_cfg['start_epoch'] * steps_per_epoch):]

    # Optimizer
    loss_scale_manager = FixedLossScaleManager(experiment_cfg['loss_scale'], False)
    opt = nn.Adam(get_param_groups(net),
                  learning_rate=Tensor(lr),
                  loss_scale=experiment_cfg['loss_scale'],
                  weight_decay=experiment_cfg['weight_decay'])

    # Create model
    model = Model(network=net, optimizer=opt, loss_scale_manager=loss_scale_manager)

    # Callbacks
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * experiment_cfg['checkpoint_interval'],
                                   keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix=model_name,
                              directory=os.path.join(local_train_url, experiment_cfg['experiment_tag']),
                              config=ckpt_config)
    if args.run_distribute:
        if get_rank() == 0:
            callbacks = [TimeLossMonitor(lr_init=lr), ckpt_cb]
        else:
            callbacks = [TimeLossMonitor(lr_init=lr)]
    else:
        callbacks = [TimeLossMonitor(lr_init=lr), ckpt_cb]

    # Training
    model.train(model_cfg.total_epochs, loader, callbacks=callbacks, dataset_sink_mode=True)

    last_checkpoint = os.path.join(local_train_url, experiment_cfg['experiment_tag'], f"{model_name}-final.ckpt")
    save_checkpoint(net, last_checkpoint)


if __name__ == "__main__":
    main()
