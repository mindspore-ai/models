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
"""evaluation"""
import os
import numpy as np
import pandas as pd
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.net import SBNetWork
from src.logger import get_logger
from src.model_utils.config import config
from src.dataloader import minddataset_loader_val
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    pass


def get_data_size(csv_path):
    csv_path = os.path.join(csv_path, './ds_size.csv')
    result = pd.read_csv(csv_path)
    val_size = result['size'][1]
    return val_size


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=config.device_id)
    network = SBNetWork(in_chanel=[19, 64, 128],
                        out_chanle=config.conv_channels,
                        dense_size=config.dense_sizes,
                        lmbda=config.lmbda,
                        isize=config.isize, keep_prob=1.0)
    network.set_train(False)
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(network, param_dict)
    val_size = get_data_size(config.mindrecord_path)
    val_path = './val/validation_dataset.mindrecord'
    val_rotation_data, val_rot_weight = minddataset_loader_val(configs=config,
                                                               mindfile=os.path.join(config.mindrecord_path, val_path),
                                                               no_batch_size=val_size)
    val_data_loader = val_rotation_data.create_dict_iterator()
    val_data_size = val_rotation_data.get_dataset_size()
    config.logger.info("Finish Load dataset and Network. dataset size: validation %d", val_data_size)
    final_mse_v = 0
    for _, vdata in enumerate(val_data_loader):
        coord_features = vdata['coords_features']
        affinity = vdata['affinitys']
        mse_v = network(coord_features, affinity)
        temp_mse_v = mse_v.asnumpy() * val_rot_weight
        final_mse_v += temp_mse_v
    config.logger.info('Validation RMSE: [%.2f]', np.sqrt(final_mse_v))


if __name__ == '__main__':
    config.logger = get_logger('./', config.device_id)
    config.logger.save_args(config)
    run_eval()
