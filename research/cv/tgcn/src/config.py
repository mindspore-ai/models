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
Configuration of parameters

For detailed information, please refer to the paper below:
https://arxiv.org/pdf/1811.05320.pdf
"""


class ConfigTGCN:
    """
    Class of parameters configuration
    """

    # Choose device: ['Ascend', 'GPU']
    device = 'Ascend'
    # Global random seed
    seed = 1

    # Use 'save_best = True' for saving model checkpoints with best RMSE on evaluation dataset
    # WARNING! Will decrease performance due to the need of evaluation after each epoch
    save_best = True

    # Dataset absolute path
    data_path = './data/'

    # Choose datasets: ['SZ-taxi', 'Los-loop', etc]
    dataset = 'SZ-taxi'

    # hidden_dim: 100 for 'SZ-taxi'; 64 for 'Los-loop'
    hidden_dim = 100
    # seq_len: 4 for 'SZ-taxi'; 12 for 'Los-loop'
    seq_len = 4
    # pre_len: [1, 2, 3, 4] separately for 'SZ-taxi'; [3, 6, 9, 12] separately for 'Los-loop'
    pre_len = 1

    # Training parameters
    train_split_rate = 0.8
    epochs = 3000
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 1.5e-3
    data_sink = True
