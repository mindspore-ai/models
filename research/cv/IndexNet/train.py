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
"""Training script."""
import numpy as np
from mindspore import Model
from mindspore import Parameter
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import nn
from mindspore import ops
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor

from src.cfg.config import config as default_config
from src.dataset import ImageMattingDatasetTrain
from src.model import LossWrapper
from src.model import MobileNetV2UNetDecoderIndexLearning
from src.utils import weighted_loss


def set_lr(cfg, steps_per_epoch):
    """
    Set lr for each step of training.

    Args:
        cfg: Config parameters.
        steps_per_epoch (int): Number of batches into one epoch on one device.

    Returns:
        lr_each_step (np.array): Learning rate for every step during training.
    """
    base_lr = cfg.learning_rate
    total_steps = int(cfg.epochs * steps_per_epoch)
    milestone_1 = cfg.milestones[0]
    milestone_2 = cfg.milestones[1]
    lr_decay = cfg.lr_decay

    lr_each_step = []
    for i in range(total_steps):
        if i < steps_per_epoch * milestone_1:
            lr5 = base_lr
        elif steps_per_epoch * milestone_1 <= i < steps_per_epoch * (milestone_1 + 1):
            lr5 = base_lr * lr_decay * 0.1
        elif steps_per_epoch * (milestone_1 + 1) <= i < steps_per_epoch * milestone_2:
            lr5 = base_lr * lr_decay
        elif steps_per_epoch * milestone_2 <= i < steps_per_epoch * (milestone_2 + 1):
            lr5 = base_lr * lr_decay ** 2 * 0.1
        elif steps_per_epoch * (milestone_2 + 1) <= i:
            lr5 = base_lr * lr_decay ** 2

        lr_each_step.append(lr5)

    return np.array(lr_each_step, np.float32)


def load_pretrained(network, backbone_names, other_names, ckpt_url):
    """
    Load weights from pretrained backbone.

    Args:
        network: Inited model.
        backbone_names (list): Backbone parameters names.
        other_names (list): Network parameters names except backbone names.
        ckpt_url (str): Path to pretrained backbone checkpoint.
    """
    model_inited_params = dict(network.parameters_and_names())
    mobilenet_params = load_checkpoint(ckpt_url, filter_prefix=['moments', 'head'])
    mobilenet_names = list(mobilenet_params.keys())[:-3]
    clear_names = []
    for name in mobilenet_names:
        if name.startswith('features.18'):
            continue
        clear_names.append(name)

    strict_names = []
    for name in clear_names:
        if name.endswith('beta'):
            strict_names.append(name.replace('beta', 'moving_mean'))
            strict_names.append(name.replace('beta', 'moving_variance'))
            strict_names.append(name.replace('beta', 'gamma'))
            strict_names.append(name)
        elif name.endswith('weight'):
            strict_names.append(name)

    state_dict = {}
    for net_name, mobil_name in zip(backbone_names, strict_names):
        weight = mobilenet_params[mobil_name][:]
        if mobil_name == 'features.0.features.0.weight':
            expand_weight = ops.Zeros()((32, 1, 3, 3), mstype.float32)
            weight = ops.Concat(axis=1)((weight, expand_weight))

        model_param = Parameter(weight, name=net_name)

        state_dict[net_name] = model_param

    for name in other_names:
        state_dict[name] = model_inited_params[name]

    load_param_into_net(network, state_dict, strict_load=False)


def set_context(cfg):
    """
    Set process context.

    Args:
        cfg: Config parameters.

    Returns:
        dev_target (str): Device target platform.
        dev_num (int): Amount of devices participating in process.
        dev_id (int): Current process device id..
    """
    dev_target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=dev_target)

    if dev_target == 'GPU':
        if cfg.is_distributed:
            init(backend_name='nccl')
            dev_num = get_group_size()
            dev_id = get_rank()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=dev_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
        else:
            dev_num = 1
            dev_id = cfg.device_id
            context.set_context(device_id=dev_id)
    else:
        raise ValueError("Unsupported platform.")

    return dev_num, dev_id


def init_callbacks(cfg, batch_number, dev_id, network):
    """
    Initialize training callbacks.

    Args:
        cfg: Config parameters.
        batch_number: Number of batches into one epoch on one device.
        dev_id: Current process device id.
        network: Network to be save into checkpoint.

    Returns:
        cbs: Inited callbacks.
    """
    loss_cb = LossMonitor(per_print_times=100)
    time_cb = TimeMonitor(data_size=batch_number)

    if cfg.is_distributed and dev_id != cfg.device_start:
        cbs = [loss_cb, time_cb]
    else:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=batch_number,
            keep_checkpoint_max=cfg.keep_checkpoint_max,
            saved_network=network,
        )

        ckpt_cb = ModelCheckpoint(
            prefix="IndexNet",
            directory=cfg.logs_dir,
            config=config_ck,
        )

        cbs = [loss_cb, time_cb, ckpt_cb]

    return cbs


def train(config):
    """
    Init model, dataset, run training.

    Args:
        config: Config parameters.
    """
    rank_size, rank_id = set_context(config)

    data = ImageMattingDatasetTrain(
        data_dir=config.data_dir,
        bg_dir=config.bg_dir,
        config=config,
        sub_folder='train',
        data_file='data.txt',
    )

    net = MobileNetV2UNetDecoderIndexLearning(
        encoder_rate=config.rate,
        encoder_current_stride=config.current_stride,
        encoder_settings=config.inverted_residual_setting,
        output_stride=config.output_stride,
        width_mult=config.width_mult,
        conv_operator=config.conv_operator,
        decoder_kernel_size=config.decoder_kernel_size,
        apply_aspp=config.apply_aspp,
        use_nonlinear=config.use_nonlinear,
        use_context=config.use_context,
    )

    net_with_loss = LossWrapper(model=net, loss_function=weighted_loss)
    net_with_loss.set_train(True)

    dataloader = GeneratorDataset(
        source=data,
        column_names=['image', 'mask', 'alpha', 'fg', 'bg', 'c_g'],
        shuffle=True,
        num_parallel_workers=config.num_workers,
        python_multiprocessing=True,
        num_shards=rank_size,
        shard_id=rank_id,
    )

    dataloader = dataloader.batch(config.batch_size, True)
    batch_num = dataloader.get_dataset_size()

    pretrained_params = []
    pretrained_names = []
    learning_params = []
    learning_names = []
    for p in net_with_loss.parameters_and_names():
        if 'dconv' in p[0] or 'pred' in p[0] or 'index' in p[0]:
            if p[1].requires_grad:
                learning_params.append(p[1])
            learning_names.append(p[0])
        else:
            if p[1].requires_grad:
                pretrained_params.append(p[1])
            pretrained_names.append(p[0])

    load_pretrained(net_with_loss, pretrained_names, learning_names, config.ckpt_url)

    lr_steps = set_lr(config, batch_num)

    opt = nn.Adam(
        [
            {'params': learning_params, 'lr': lr_steps},
            {'params': pretrained_params, 'lr': lr_steps / config.backbone_lr_mult},
        ],
        learning_rate=lr_steps
    )

    model = Model(net_with_loss, optimizer=opt)

    callbacks = init_callbacks(config, batch_num, rank_id, net)

    model.train(epoch=config.epochs, train_dataset=dataloader, callbacks=callbacks, dataset_sink_mode=False)
    print("train success")


if __name__ == '__main__':
    set_seed(1)
    train(config=default_config)
