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
Export checkpoints into MINDIR model files
"""
import os
import argparse
import numpy as np
from mindspore import export, load_checkpoint, load_param_into_net, Tensor, context
from src.config import ConfigTGCN
from src.task import SupervisedForecastTask
from src.dataprocess import load_adj_matrix


# Set DEVICE_ID
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', help="DEVICE_ID", type=int, default=0)
args = parser.parse_args()


if __name__ == '__main__':
    # Config initialization
    config = ConfigTGCN()
    # Runtime
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device, device_id=args.device_id)
    # Create network
    adj = (load_adj_matrix(config.dataset))
    net = SupervisedForecastTask(adj, config.hidden_dim, config.pre_len)
    # Load parameters from checkpoint into network
    file_name = config.dataset + "_" + str(config.pre_len) + ".ckpt"
    param_dict = load_checkpoint(os.path.join('checkpoints', file_name))
    load_param_into_net(net, param_dict)
    # Initialize dummy inputs
    inputs = np.random.uniform(0.0, 1.0, size=[config.batch_size, config.seq_len, adj.shape[0]]).astype(np.float32)
    # Export network into MINDIR model file
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    file_name = config.dataset + "_" + str(config.pre_len)
    path = os.path.join('outputs', file_name)
    export(net, Tensor(inputs), file_name=path, file_format='MINDIR')
    print("==========================================")
    print(file_name + ".mindir exported successfully!")
    print("==========================================")
