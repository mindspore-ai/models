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
Evaluation script
"""
import os
import argparse
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from src.config import ConfigTGCN
from src.task import SupervisedForecastTask
from src.dataprocess import load_adj_matrix, load_feat_matrix, generate_dataset_np
from src.metrics import evaluate_network


# Set DEVICE_ID
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', help="DEVICE_ID", type=int, default=0)
parser.add_argument('--data_path', help="directory of datasets", type=str, default='./data')
args = parser.parse_args()


if __name__ == '__main__':
    # Config initialization
    config = ConfigTGCN()
    # Runtime
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device, device_id=args.device_id)
    # Create network
    net = SupervisedForecastTask(load_adj_matrix(config.dataset, config.data_path), config.hidden_dim, config.pre_len)
    # Load parameters from checkpoint into network
    ckpt_file_name = config.dataset + "_" + str(config.pre_len) + ".ckpt"
    param_dict = load_checkpoint(os.path.join('checkpoints', ckpt_file_name))
    load_param_into_net(net, param_dict)
    # Evaluation
    feat, max_val = load_feat_matrix(config.dataset, config.data_path)
    _, _, eval_inputs, eval_targets = generate_dataset_np(feat, config.seq_len, config.pre_len, config.train_split_rate)
    evaluate_network(net, max_val, eval_inputs, eval_targets)
