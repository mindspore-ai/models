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
'''
train
'''
from __future__ import division

import os

import numpy as np
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint,\
                                     CheckpointConfig, SummaryCollector
from mindspore.nn.optim import Adam
from mindspore.common import set_seed

from src.dataset import CreateDatasetCoco
from src.config import config
from src.network_with_loss import JointsMSELoss, PoseResNetWithLoss
from src.FastPose import createModel


def get_lr(begin_epoch,
           total_epochs,
           steps_per_epoch,
           lr_init=0.001,
           factor=0.1,
           epoch_number_to_drop=(170, 200)
           ):
    '''
    get_lr
    '''
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    step_number_to_drop = [steps_per_epoch * x for x in epoch_number_to_drop]
    for i in range(int(total_steps)):
        if i in step_number_to_drop:
            lr_init = lr_init * factor
        lr_each_step.append(lr_init)
    current_step = steps_per_epoch * begin_epoch
    lr_each_step = np.array(lr_each_step, dtype=np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def main():
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.DEVICE_TARGET,
                        save_graphs=False)

    if config.DEVICE_TARGET == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)

    if config.RUN_DISTRIBUTE:
        if config.DEVICE_TARGET == 'Ascend':
            init()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              parameter_broadcast=True)
        elif config.DEVICE_TARGET == 'GPU':
            init("nccl")
            rank = get_rank()
            device_num = get_group_size()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            raise NotImplementedError("Only GPU and Ascend training supported")
    else:
        rank = 0
        device_num = 1

    if config.MODELARTS_IS_MODEL_ARTS:
        mox.file.copy_parallel(src_url=config.MODELARTS_DATA_URL,
                               dst_url=config.MODELARTS_CACHE_INPUT)

    print(f"Running on {config.DEVICE_TARGET}, device num: {device_num}, rank: {rank}")
    dataset = CreateDatasetCoco(rank=rank,
                                group_size=device_num,
                                train_mode=True,
                                num_parallel_workers=config.TRAIN_NUM_PARALLEL_WORKERS,
                                )
    m = createModel()
    loss = JointsMSELoss(config.LOSS_USE_TARGET_WEIGHT)
    net_with_loss = PoseResNetWithLoss(m, loss)
    dataset_size = dataset.get_dataset_size()
    print(f"Dataset size = {dataset_size}")
    lr = Tensor(get_lr(config.TRAIN_BEGIN_EPOCH,
                       config.TRAIN_END_EPOCH,
                       dataset_size,
                       lr_init=config.TRAIN_LR,
                       factor=config.TRAIN_LR_FACTOR,
                       epoch_number_to_drop=config.TRAIN_LR_STEP))
    optim = Adam(m.trainable_params(), learning_rate=lr)
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor()
    summary_cb = SummaryCollector(os.path.join(config.SUMMARY_DIR, f'rank_{rank}'))
    cb = [time_cb, loss_cb, summary_cb]
    if config.TRAIN_SAVE_CKPT:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=dataset_size, keep_checkpoint_max=2)
        prefix = ''
        if config.RUN_DISTRIBUTE:
            prefix = 'multi_' + 'train_fastpose_' + \
                config.VERSION + '_' + str(rank)
        else:
            prefix = 'single_' + 'train_fastpose_' + config.VERSION

        directory = ''
        if config.MODELARTS_IS_MODEL_ARTS:
            directory = config.MODELARTS_CACHE_OUTPUT + \
                'device_' + str(rank)
        elif config.RUN_DISTRIBUTE:
            directory = config.TRAIN_CKPT_PATH + \
                'device_' + str(rank)
        else:
            directory = config.TRAIN_CKPT_PATH + 'device'

        ckpoint_cb = ModelCheckpoint(
            prefix=prefix, directory=directory, config=config_ck)
        cb.append(ckpoint_cb)
    model = Model(net_with_loss, optimizer=optim, amp_level="O2")
    epoch_size = config.TRAIN_END_EPOCH - config.TRAIN_BEGIN_EPOCH

    print("************ Start training now ************")
    print(f'start training, {epoch_size} epochs, {dataset_size} steps per epoch')
    model.train(epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)

    if config.MODELARTS_IS_MODEL_ARTS:
        mox.file.copy_parallel(
            src_url=config.MODELARTS_CACHE_OUTPUT, dst_url=config.MODELARTS_TRAIN_URL)


if __name__ == '__main__':
    if config.MODELARTS_IS_MODEL_ARTS:
        import moxing as mox
    set_seed(config.TRAIN_SEED)

    main()
