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
"""Train HiFaceGAN model"""
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode

from src.dataset.dataset import create_train_dataset
from src.model.cell import TrainOneStepD
from src.model.cell import TrainOneStepG
from src.model.discriminator import MultiscaleDiscriminator
from src.model.generator import HiFaceGANGenerator
from src.model.initializer import init_weights
from src.model.loss import DiscriminatorLoss
from src.model.loss import GeneratorLoss
from src.model.reporter import Reporter
from src.model_utils.config import get_config
from src.util import clip_adam_param
from src.util import get_lr
from src.util import set_global_seed


def train_preprocess(config):
    """Set context before training"""
    if config.is_distributed:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=config.group_size)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)
        config.rank = 0
        config.group_size = 1


def run_train(config):
    """Train HiFaceGAN network"""
    train_preprocess(config)

    dataset, dataset_size = create_train_dataset(
        data_root=config.data_path,
        degradation_type=config.degradation_type,
        batch_size=config.batch_size,
        is_distributed=config.is_distributed,
        group_size=config.group_size,
        rank=config.rank,
        img_size=config.img_size
    )
    num_epochs = config.num_epochs + config.num_epochs_decay
    dataloader = dataset.create_dict_iterator(num_epochs=num_epochs)
    config.dataset_size = dataset_size

    if config.rank == 0:
        reporter = Reporter(config)
        reporter.info('========== start training ===============')

    generator = HiFaceGANGenerator(
        ngf=config.ngf,
        input_nc=config.input_nc
    )
    discriminator = MultiscaleDiscriminator(
        ndf=config.ndf,
        input_nc=config.input_nc,
        use_gan_feat_loss=config.use_gan_feat_loss
    )

    init_weights(generator)
    init_weights(discriminator)

    generator_loss = GeneratorLoss(
        generator=generator,
        discriminator=discriminator,
        pretrained_vgg_path=config.pretrained_vgg_path,
        use_vgg_loss=config.use_vgg_loss,
        use_gan_feat_loss=config.use_gan_feat_loss,
        lambda_feat=config.lambda_feat,
        lambda_vgg=config.lambda_vgg
    )
    discriminator_loss = DiscriminatorLoss(discriminator)

    gen_lr = config.lr / 2 if config.use_ttur else config.lr
    dis_lr = config.lr * 2 if config.use_ttur else config.lr

    generator_optim = nn.Adam(
        params=generator.trainable_params(),
        learning_rate=get_lr(initial_lr=gen_lr, lr_policy=config.lr_policy, num_epochs=config.num_epochs,
                             num_epochs_decay=config.num_epochs_decay, dataset_size=config.dataset_size),
        beta1=clip_adam_param(config.beta1),
        beta2=clip_adam_param(config.beta2)
    )
    discriminator_optim = nn.Adam(
        params=discriminator_loss.trainable_params(),
        learning_rate=get_lr(initial_lr=dis_lr, lr_policy=config.lr_policy, num_epochs=config.num_epochs,
                             num_epochs_decay=config.num_epochs_decay, dataset_size=config.dataset_size),
        beta1=clip_adam_param(config.beta1),
        beta2=clip_adam_param(config.beta2)
    )
    net_G = TrainOneStepG(generator_loss, generator, generator_optim)
    net_D = TrainOneStepD(discriminator_loss, discriminator_optim)
    net_G.set_train()
    net_D.set_train()

    for _ in range(num_epochs):
        if config.rank == 0:
            reporter.epoch_start()
        for data in dataloader:
            lq = data['low_quality']
            hq = data['high_quality']
            *res_G, generated = net_G(lq, hq)
            res_D = net_D(lq, hq, generated)
            if config.rank == 0:
                reporter.step_end(res_G, res_D)
                reporter.visualizer(lq, hq, generated)
        if config.rank == 0:
            reporter.epoch_end(net_G)

    if config.rank == 0:
        reporter.info('========== end training ===============')


if __name__ == '__main__':
    set_global_seed(0)
    cfg = get_config()
    run_train(cfg)
