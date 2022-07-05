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
"""model training"""
import os
import time
import numpy as np
import pandas as pd
from mindspore import nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.common import set_seed
from mindspore.nn import TrainOneStepCell
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.serialization import save_checkpoint

from src.logger import get_logger
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.net import SBNetWork
from src.dataloader import minddataset_loader, minddataset_loader_val

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

set_seed(123)

def modelarts_pre_process():
    pass

def get_data_size(csv_path):
    csv_path = os.path.join(csv_path, './ds_size.csv')
    result = pd.read_csv(csv_path)
    train_size = result['size'][0]
    val_size = result['size'][1]
    return train_size, val_size


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    config.logger = get_logger('./', config.device_id)
    config.logger.save_args(config)
    device_num = get_device_num()
    if device_num > 1:
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        context.set_context(device_id=config.device_id)

    network = SBNetWork(in_chanel=[19, 64, 128],
                        out_chanle=config.conv_channels,
                        dense_size=config.dense_sizes,
                        lmbda=config.lmbda,
                        isize=config.isize, keep_prob=config.keep_prob)
    lr = Tensor(float(config.lr))
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr, weight_decay=config.weight_decay)
    train_wrapper = TrainOneStepCell(network, optimizer=optimizer)
    train_size, val_size = get_data_size(config.mindrecord_path)
    rot_path = './train_rotation/train_rotation_dataset.mindrecord'
    nrot_path = './no_rotation/train_norotation_dataset.mindrecord'
    val_path = './val/validation_dataset.mindrecord'
    rotation_data, _ = minddataset_loader(configs=config,
                                          mindfile=os.path.join(config.mindrecord_path, rot_path),
                                          no_batch_size=train_size)

    no_rotation_data, no_rot_weight = minddataset_loader(configs=config,
                                                         mindfile=os.path.join(config.mindrecord_path, nrot_path),
                                                         no_batch_size=train_size)
    val_rotation_data, val_rot_weight = minddataset_loader_val(configs=config,
                                                               mindfile=os.path.join(config.mindrecord_path, val_path),
                                                               no_batch_size=val_size)
    config.logger.info('train weight: %s, validation weight:%s.', no_rot_weight, val_rot_weight)
    config.logger.info("Finish Load dataset and Network. ")
    rot_data_loader = rotation_data.create_dict_iterator()
    rot_data_size = rotation_data.get_dataset_size()

    norot_data_loader = no_rotation_data.create_dict_iterator()
    norot_data_size = no_rotation_data.get_dataset_size()

    val_data_loader = val_rotation_data.create_dict_iterator()
    val_data_size = val_rotation_data.get_dataset_size()
    config.logger.info("Finish Load dataset and Network. dataset size: rotation %d, "
                       "without rotation %d, val %d", rot_data_size, norot_data_size, val_data_size)
    stand_mse_v = float('inf')
    per_step_time = 0
    for epoch in range(config.epoch_size):
        # with rotation
        fianl_mse = 0
        final_mse_v = 0
        for _, data in enumerate(rot_data_loader):
            coord_features = data['coords_features']
            affinity = data['affinitys']
            train_wrapper(coord_features, affinity, False)
        config.logger.info("Starting No Ratation.....")
        for _, datanr in enumerate(norot_data_loader):
            coord_features = datanr['coords_features']
            affinity = datanr['affinitys']
            time_start = time.time()
            nrot_mse = train_wrapper(coord_features, affinity, True)
            time_end = time.time()
            per_step_time = time_end - time_start
            fianl_mse = nrot_mse.asnumpy()
            fianl_mse *= no_rot_weight
        # validation
        for _, vdata in enumerate(val_data_loader):
            coord_features = vdata['coords_features']
            affinity = vdata['affinitys']
            mse_v = network(coord_features, affinity)
            temp_mse_v = mse_v.asnumpy() * val_rot_weight
            final_mse_v += temp_mse_v
        config.logger.info('epoch[%d], train error: [%s], '
                           'Validation error: [%s], Validation RMSE: [%s], per_step_time: [%s]',
                           epoch, fianl_mse.sum(), final_mse_v, np.sqrt(final_mse_v), per_step_time*1000)
        if final_mse_v <= stand_mse_v:
            stand_mse_v = final_mse_v
            config.logger.info("Saving checkpoint file")
            save_checkpoint(train_wrapper, f'./ckpt/pafnucy_{final_mse_v}_{epoch}.ckpt')

    config.logger.info("Finish Training.....")


if __name__ == '__main__':
    run_train()
