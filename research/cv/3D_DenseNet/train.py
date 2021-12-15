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
"""
Train net module
"""
import os
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore import set_seed
from src.loss import SoftmaxCrossEntropyWithLogits
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.dataloader import H5Dataset
from src.model import DenseNet
from src.eval_call_back import StepLossAccInfo

set_seed(1)

@moxing_wrapper()
def train_net():
    """
    Mindspore trianing net definition
    """
    if config.device_target == 'Ascend':
        device_id = int(os.getenv('DEVICE_ID'), 0)
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, \
                        device_id=device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    print(config.run_distribute)
    if config.run_distribute:
        init()
        if config.device_target == 'Ascend':
            rank_id = get_device_id()
            rank_size = get_device_num()
        else:
            rank_id = get_rank()
            rank_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=rank_size,
                                          gradients_mean=False)
    else:
        rank_id = 0
        rank_size = 1
    mri_data_train = H5Dataset(config.train_dir, mode='train')
    trainds = ds.GeneratorDataset(mri_data_train, ["data", "label"], shuffle=True)
    trainds = trainds.batch(config.batch_size, drop_remainder=True)
    train_data_size = trainds.get_dataset_size()
    mri_data_val = H5Dataset(config.val_dir, mode='val')
    valds = ds.GeneratorDataset(mri_data_val, ["data", "label"], shuffle=False)
    valds = valds.batch(config.val_batch)
    print("train dataset length is:", train_data_size)
    network = DenseNet(num_init_features=config.num_init_features, growth_rate=config.growth_rate,\
                block_config=config.block_config, drop_rate=config.drop_rate, num_classes=config.num_classes)
    loss = SoftmaxCrossEntropyWithLogits()
    step_size_S = config.save_checkpoint_steps
    lr_S = config.lr
    nn.piecewise_constant_lr(milestone=[step_size_S, 2*step_size_S, 3*step_size_S, 4*step_size_S],\
                                    learning_rates=[0.1*lr_S, 0.01*lr_S, 0.001*lr_S, 0.0001*lr_S])
    optimizer = nn.SGD(params=network.trainable_params(), learning_rate=lr_S)
    network.set_train()
    if config.device_target == 'GPU' and config.enable_fp16_gpu:
        model = Model(network, loss_fn=loss, optimizer=optimizer, amp_level='O2')
    else:
        model = Model(network, loss_fn=loss, optimizer=optimizer)
    time_cb = TimeMonitor(data_size=train_data_size)
    loss_cb = LossMonitor()
    steps_loss = {"step": [], "loss_value": []}
    steps_eval = {"step": [], "dice": []}
    step_loss_acc_info = StepLossAccInfo(network, valds, steps_loss, steps_eval)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    ckpoint_cb = ModelCheckpoint(prefix='3D-DenseSeg',
                                 directory=ckpt_save_dir + './ckpt_{}/'.format(rank_id),
                                 config=ckpt_config)
    callbacks_list = [loss_cb, time_cb, ckpoint_cb, step_loss_acc_info]
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset=trainds, callbacks=callbacks_list)
    print("============== End Training ==============")

if __name__ == '__main__':
    train_net()
