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
Train TransE/TransD/TransH/TransR models
"""
import datetime
import os

from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim.sgd import SGD
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from src.dataset import create_train_dataset
from src.loss import LossWrapperCell
from src.loss import TripletsMarginLoss
from src.model_builder import create_model
from src.utils.logging import get_logger

set_seed(config.seed)


def modelarts_pre_process():
    """modelarts pre process function."""


def _prepare_context():
    """Prepare the MindSpore context"""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    if config.is_train_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()

        context.set_context(device_id=config.rank)

        device_num = config.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
        )
    else:
        config.rank = 0
        config.group_size = 1
        context.set_context(device_id=config.device_id)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """Run train"""
    # Prepare the Context (number of devices, their type and IDs)
    _prepare_context()

    # Determine, do we need to save all checkpoints or only for the process with Rank=0
    save_ckpt_flag = False
    if config.ckpt_save_on_master_only:
        if config.rank == 0:
            save_ckpt_flag = True
    else:
        save_ckpt_flag = True

    # logger
    time_label = datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S')
    config.train_output_dir = os.path.join(config.train_output_dir, f"{time_label}_{config.model_name}")
    config.logger = get_logger(config.train_output_dir, config.rank)

    dataset, ent_tot, rel_tot = create_train_dataset(
        dataset_root=config.dataset_root,
        triplet_file_name=config.train_triplet_file_name,
        entities_file_name=config.entities_file_name,
        relations_file_name=config.relations_file_name,
        negative_sampling_rate=config.negative_sampling_rate,
        batch_size=config.train_batch_size,
        group_size=config.group_size,
        rank=config.rank,
        seed=config.seed,
    )

    batch_num = dataset.get_dataset_size()
    config.steps_per_epoch = dataset.get_dataset_size()
    config.logger.save_args(config)

    # network
    config.logger.important_info('start create network')

    # get network and init
    network = create_model(ent_tot, rel_tot, config)

    # pre_trained
    if config.pre_trained:
        load_param_into_net(network, load_checkpoint(config.pre_trained))

    loss = TripletsMarginLoss(config.negative_sampling_rate, config.margin)

    # optimizer
    opt = SGD(
        params=network.trainable_params(),
        learning_rate=Tensor(config.lr, mstype.float32),
        weight_decay=config.weight_decay,
    )

    network_with_loss = LossWrapperCell(network, loss)
    model = Model(network_with_loss, optimizer=opt)

    # define callbacks
    callbacks = [
        TimeMonitor(data_size=batch_num),
        LossMonitor(per_print_times=batch_num),
    ]
    if save_ckpt_flag:
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=config.ckpt_save_interval * config.steps_per_epoch,
            keep_checkpoint_max=config.keep_checkpoint_max,
            saved_network=network,
        )
        save_ckpt_path = os.path.join(config.train_output_dir, 'ckpt_' + str(config.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=save_ckpt_path, prefix='{}'.format(config.rank))
        callbacks.append(ckpt_cb)

    model.train(config.epochs_num, dataset, callbacks=callbacks, dataset_sink_mode=config.train_use_data_sink)


if __name__ == '__main__':
    run_train()
