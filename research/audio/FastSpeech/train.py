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
import os

import numpy as np
from mindspore import Model
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.dataset import GeneratorDataset
from mindspore.nn import Adam
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor

from src.cfg.config import config as default_config
from src.dataset import BufferDataset
from src.dataset import get_data_to_buffer
from src.model import FastSpeech
from src.model import LossWrapper

set_seed(1)


def _get_rank_info(target):
    """
    Get rank size and rank id.
    """
    if target == 'GPU':
        num_devices = get_group_size()
        device = get_rank()
    else:
        raise ValueError("Unsupported platform.")

    return num_devices, device


def lr_scheduler(cfg, steps_per_epoch, p_num):
    """
    Init lr steps.
    """
    d_model = cfg.decoder_dim
    lr_init = np.power(d_model, -0.5) * cfg.lr_scale
    warmup_steps = cfg.n_warm_up_step
    total_steps = cfg.epochs * steps_per_epoch

    learning_rate = []
    for step in range(1, total_steps + 1):
        lr_at_step = np.min([
            np.power(step * p_num, -0.5),
            np.power(warmup_steps, -1.5) * step
        ])
        learning_rate.append(lr_at_step * lr_init)

    return learning_rate


def set_trainable_params(params):
    """
    Freeze positional encoding layers
    and exclude it from trainable params for optimizer.
    """
    trainable_params = []
    for param in params:
        if param.name.endswith('position_enc.embedding_table'):
            param.requires_grad = False
        else:
            trainable_params.append(param)

    return trainable_params


def main():
    """Trainloop."""
    config = default_config
    device_target = config.device_target

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    device_num = int(os.getenv('RANK_SIZE', '1'))

    if device_target == 'GPU':
        if device_num > 1:
            init(backend_name='nccl')
            device_num = get_group_size()
            device_id = get_rank()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
        else:
            device_num = 1
            device_id = config.device_id
            context.set_context(device_id=device_id)
    else:
        raise ValueError("Unsupported platform.")

    if device_num > 1:
        rank_size, rank_id = _get_rank_info(target=device_target)
    else:
        rank_size, rank_id = None, None

    net = FastSpeech()
    network = LossWrapper(net)
    network.set_train(True)

    buffer = get_data_to_buffer()
    data = BufferDataset(buffer)

    dataloader = GeneratorDataset(
        data,
        column_names=['text', 'mel_pos', 'src_pos', 'mel_max_len', 'duration', 'mel_target'],
        shuffle=True,
        num_shards=rank_size,
        shard_id=rank_id,
        num_parallel_workers=1,
        python_multiprocessing=False,
    )

    dataloader = dataloader.batch(config.batch_size, True)
    batch_num = dataloader.get_dataset_size()

    lr = lr_scheduler(config, batch_num, device_num)

    trainable_params = set_trainable_params(network.trainable_params())
    opt = Adam(trainable_params, beta1=0.9, beta2=0.98, eps=1e-9, learning_rate=lr)

    model = Model(network, optimizer=opt)

    config_ck = CheckpointConfig(
        save_checkpoint_steps=batch_num,
        keep_checkpoint_max=config.keep_checkpoint_max,
    )

    loss_cb = LossMonitor(per_print_times=10)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpt_cb = ModelCheckpoint(
        prefix="FastSpeech",
        directory=config.logs_dir,
        config=config_ck,
    )

    cbs = [loss_cb, time_cb, ckpt_cb]
    if device_num > 1 and device_id != config.device_start:
        cbs = [loss_cb, time_cb]

    model.train(epoch=config.epochs, train_dataset=dataloader, callbacks=cbs, dataset_sink_mode=False)


if __name__ == "__main__":
    main()
