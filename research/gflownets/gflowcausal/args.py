# Copyright 2023 Huawei Technologies Co., Ltd
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
config
"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--device_platform", default='GPU', type=str, choices=['GPU', 'CPU'])
parser.add_argument("--seed", default=1111, type=int)
parser.add_argument("--learning_rate", default=5e-4, help="Learning rate", type=float)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=1e-3, type=float)

parser.add_argument("--mbsize", default=64, help="Minibatch size", type=int)  # batch size
parser.add_argument("--epoches", default=3000, help=" epoches", type=int)  # epoch
parser.add_argument("--thresh", default=0.3, help=" thresh", type=int)
parser.add_argument("--train_to_sample_ratio", default=1, type=float)
parser.add_argument("--n_hid", default=256, type=int)  # MLP
parser.add_argument("--n_layers", default=3, type=int)  # MLP

parser.add_argument("--embed_dim", default=512, type=int)
parser.add_argument("--num_heads", default=128, type=int)
parser.add_argument("--sampling_size", default=500, type=int)
parser.add_argument("--regression_type", default='linear', type=str)
parser.add_argument("--score_type", default='BIC', type=str)
parser.add_argument("--n_node", default=30, type=int)
# Data
parser.add_argument("--n_edges", default=60, type=int)
parser.add_argument("--n_samples", default=1000, type=int)
parser.add_argument("--sem_type", default='linear', type=str)
parser.add_argument("--reg_type", default='gauss', type=str)
parser.add_argument("--data_scheme", default='ER', type=str, choices=['ER', 'SF'])  # Simulation data
parser.add_argument("--save_dir", default='save_models/BIC_{}_{}_nodes{}_epoch{}.pkl', type=str)
# Flownet
parser.add_argument("--bootstrap_tau", default=0., type=float)
parser.add_argument("--replay_strategy", default='3', type=str)  # top_k no
# ne
parser.add_argument("--replay_sample_size", default=2, type=int)
parser.add_argument("--replay_buf_size", default=0, type=float)
parser.add_argument("--process_num", default=8, type=int)
parser.add_argument("--model_name", default='MLP', type=str, choices=['DNN', 'CNN', 'MLP', 'MLP_encode'])  # MLP,DNN,CNN

args = parser.parse_args()
