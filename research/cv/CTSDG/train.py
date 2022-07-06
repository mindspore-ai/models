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
"""train"""

import os

from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import nn
from mindspore import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init

from model_utils.config import config
from src.callbacks import get_callbacks
from src.dataset import create_ctsdg_dataset
from src.discriminator.discriminator import Discriminator
from src.generator.generator import Generator
from src.generator.vgg16 import get_feature_extractor
from src.losses import DWithLossCell
from src.losses import GWithLossCell
from src.trainer import CTSDGTrainer
from src.trainer import DTrainOneStepCell
from src.trainer import GTrainOneStepCell
from src.utils import check_args


def set_default():
    """set default"""
    set_seed(config.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    # init distributed
    if config.is_distributed:
        init('nccl')
        config.rank = get_rank()
        config.device_num = get_group_size()
        context.reset_auto_parallel_context()
        parallel_mode = context.ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_num=config.device_num)
    else:
        config.rank = 0
        config.device_num = 1
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)

    config.save_ckpt_logs = config.rank == 0


def prepare_train(finetune):
    """prepare train"""
    generator = Generator(
        image_in_channels=config.image_in_channels,
        edge_in_channels=config.edge_in_channels,
        out_channels=config.out_channels
    )

    discriminator = Discriminator(
        image_in_channels=config.image_in_channels,
        edge_in_channels=config.edge_in_channels
    )

    if finetune:
        gen_pretrain_path = os.path.join(config.save_path, f'generator_{config.total_steps:06d}.ckpt')
        dis_pretrain_path = os.path.join(config.save_path, f'discriminator_{config.total_steps:06d}.ckpt')
        load_param_into_net(generator, load_checkpoint(gen_pretrain_path))
        load_param_into_net(discriminator, load_checkpoint(dis_pretrain_path))
        config.total_steps = config.finetune_iter
    else:
        config.total_steps = config.train_iter

    generator.set_train(not finetune)
    discriminator.set_train()

    vgg16_feat_extr = get_feature_extractor(config)
    generator_w_loss = GWithLossCell(generator, discriminator,
                                     vgg16_feat_extr, config)

    discriminator_w_loss = DWithLossCell(discriminator)

    optimizer_g = nn.Adam(generator.trainable_params(),
                          learning_rate=config.gen_lr_train)
    optimizer_d = nn.Adam(discriminator.trainable_params(),
                          learning_rate=config.dis_lr_multiplier * config.gen_lr_train)
    generator_t_step = GTrainOneStepCell(generator_w_loss,
                                         optimizer_g)
    discriminator_t_step = DTrainOneStepCell(discriminator_w_loss,
                                             optimizer_d)

    trainer = CTSDGTrainer(generator_t_step, discriminator_t_step)

    dataset = create_ctsdg_dataset(config, is_training=True)
    n_epochs = int(config.total_steps * config.device_num / config.length_dataset + 1)
    dataset = dataset.repeat(n_epochs)

    return trainer, generator, discriminator, dataset


def run_train(finetune):
    """run train"""
    trainer, generator, discriminator, dataset = prepare_train(finetune)
    dataloader = dataset.create_dict_iterator()

    if config.save_ckpt_logs:
        callbacks = get_callbacks(config, generator, discriminator, finetune)
    for num_batch, sample in enumerate(dataloader):
        if num_batch > config.total_steps:
            break
        ground_truth = sample['image']
        mask = sample['mask']
        edge = sample['edge']
        gray_image = sample['gray_image']
        loss_g, loss_d = trainer(ground_truth, mask, edge, gray_image)
        if config.save_ckpt_logs:
            callbacks([loss_g.asnumpy().mean(), loss_d.asnumpy().mean()])


if __name__ == "__main__":
    check_args(config)
    set_default()
    run_train(finetune=False)

    config.start_iter = config.total_steps
    run_train(finetune=True)
