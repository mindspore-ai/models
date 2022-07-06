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
"""Model training"""
from pathlib import Path

import numpy as np
from mindspore import Model
from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import nn
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor

from model_utils.logging import get_logger
from model_utils.moxing_adapter import config
from model_utils.util import get_param_groups
from src.bn_inception import BNInception
from src.train_cell import CustomTrainOneStepCell
from src.train_cell import CustomWithLossCell
from src.trn import RelationModuleMultiScale
from src.tsn import TSN
from src.tsn_dataset import get_dataset_for_training

set_seed(config.seed)


def initialize_backbone(backbone, checkpoint_path):
    """Initialize the BNInception backbone"""
    print("initialize backbone...")
    ckpt_data = load_checkpoint(checkpoint_path)

    # The original BN Inception has 1000 model outputs,
    # but we need config.img_feature_dim of the outputs.
    # So we just take the last fc layer from backbone.
    ckpt_data['fc.weight'] = backbone.fc.weight
    ckpt_data['fc.bias'] = backbone.fc.bias

    not_loaded = load_param_into_net(backbone, ckpt_data)
    if not_loaded:
        print(f'The following parameters are not loaded: {not_loaded}')


def get_learning_rate(lr_init, lr_decay, total_epochs, update_lr_epochs, steps_per_epoch):
    """Calculate learning rate values"""
    steps = np.arange(total_epochs * steps_per_epoch)
    epochs = steps // steps_per_epoch
    decay_steps = epochs // update_lr_epochs
    lr_values = lr_init * np.power(lr_decay, decay_steps)
    return lr_values.astype("float32")


def prepare_context(cfg):
    """Prepare context"""
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)

    if cfg.is_train_distributed:
        init()
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            device_num=cfg.group_size,
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True
        )
    else:
        cfg.rank = 0
        cfg.group_size = 1
        context.set_context(device_id=cfg.device_id)


def prepare_optimizer(cfg, network, dataset_size):
    """Prepare optimizer"""
    print("prepare optimizer...")

    # lr scheduler
    lr = get_learning_rate(
        lr_init=cfg.lr,
        lr_decay=0.1,
        total_epochs=cfg.epochs_num,
        update_lr_epochs=cfg.update_lr_epochs,
        steps_per_epoch=dataset_size,
    )

    grouped_parameters = get_param_groups(network.trainable_params())

    first_conv_weight = grouped_parameters[0]
    first_conv_bias = grouped_parameters[1]
    first_bn = grouped_parameters[2]

    normal_weight = grouped_parameters[3]
    normal_bias = grouped_parameters[4]

    optim_group_params = [
        {'params': first_conv_weight, 'lr': lr, 'weight_decay': cfg.weight_decay},
        {'params': first_conv_bias, 'lr': lr * 2, 'weight_decay': 0},
        {'params': first_bn, 'lr': lr, 'weight_decay': 0},

        {'params': normal_weight, 'lr': lr, 'weight_decay': cfg.weight_decay},
        {'params': normal_bias, 'lr': lr * 2, 'weight_decay': 0}
    ]

    optimizer = nn.SGD(params=optim_group_params, momentum=cfg.momentum)
    return optimizer


def prepare_callbacks(cfg, network, dataset_size):
    """Prepare callbacks"""
    print("prepare callbacks...")

    callbacks = [
        TimeMonitor(data_size=dataset_size),
        LossMonitor(cfg.log_interval),
    ]

    if cfg.rank == 0 or not cfg.ckpt_save_on_master_only:
        checkpoint_config = CheckpointConfig(
            save_checkpoint_steps=cfg.ckpt_save_interval * dataset_size,
            keep_checkpoint_max=cfg.keep_checkpoint_max,
            saved_network=network,
        )

        checkpoint_output_dir_path = Path(cfg.train_output_dir) / 'checkpoints'
        ckpt_save_callback = ModelCheckpoint(
            prefix="checkpoint_trn",
            config=checkpoint_config,
            directory=str(checkpoint_output_dir_path),
        )
        callbacks.append(ckpt_save_callback)

    return callbacks


def run_train(cfg):
    """Run model train"""
    prepare_context(cfg)

    logger = get_logger(cfg.train_output_dir, cfg.rank)
    logger.save_args(cfg)

    dataset, num_class = get_dataset_for_training(
        dataset_root=cfg.dataset_root,
        images_dir_name=cfg.images_dir_name,
        files_list_name=cfg.train_list_file_name,
        image_size=cfg.image_size,
        num_segments=cfg.num_segments,
        batch_size=cfg.train_batch_size,
        subsample_num=cfg.subsample_num,
        seed=cfg.seed,
        rank=cfg.rank,
        group_size=cfg.group_size,
        train_workers=cfg.train_workers,
    )

    logger.important_info('Create the model')

    # Prepare the backbone
    backbone = BNInception(out_channels=cfg.img_feature_dim, dropout=cfg.dropout, frozen_bn=True)
    initialize_backbone(backbone, cfg.pre_trained_backbone)
    trn_head = RelationModuleMultiScale(
        cfg.img_feature_dim,
        cfg.num_segments,
        num_class,
        subsample_num=cfg.subsample_num,
    )
    network = TSN(base_network=backbone, consensus_network=trn_head)

    net_optimizer = prepare_optimizer(cfg, network, dataset_size=dataset.get_dataset_size())

    # loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_loss_opt = CustomTrainOneStepCell(CustomWithLossCell(network, net_loss), net_optimizer, cfg.clip_grad_norm)

    # Callbacks
    callbacks = prepare_callbacks(cfg, network, dataset_size=dataset.get_dataset_size())

    # Creating a Model wrapper and training
    model = Model(net_loss_opt)

    logger.important_info('Train')
    logger.info('Total steps: %d', dataset.get_dataset_size())

    print("\n" + "*" * 70)
    model.train(cfg.epochs_num, dataset, callbacks=callbacks, dataset_sink_mode=True)


if __name__ == '__main__':
    run_train(config)
