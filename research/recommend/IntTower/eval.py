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

import mindspore as ms
from mindspore import nn
from model import IntTower
from util import test_epoch, setup_seed
from get_dataset import process_struct_data, construct_dataset
import model_config as cfg

if __name__ == '__main__':
    seed = 2012
    setup_seed(seed)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    batch_size = cfg.batch_size
    data_path = './data/movielens.txt'
    _, _, test_generator = process_struct_data(data_path)
    test_dataset = construct_dataset(test_generator, batch_size)
    network = IntTower()
    loss_fn = nn.BCELoss(reduction='mean')
    param_dict = ms.load_checkpoint("./IntTower.ckpt")
    ms.load_param_into_net(network, param_dict)
    test_epoch(test_dataset, network, loss_fn, test_generator, batch_size)
