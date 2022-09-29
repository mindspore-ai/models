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

import argparse
import yaml
from src.utils import load_graph_data
from src.dcrnn_supervisor import DCRNNSupervisor


def main(arg):
    with open(arg.config_filename) as f:
        print('Start Reading Config File')
        supervisor_config = yaml.safe_load(f)
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        _, _, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(arg, adj_mx=adj_mx, **supervisor_config)
        supervisor.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_la.yaml', type=str)
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--checkpoint_frequency', type=int, default=1)
    parser.add_argument('--checkpoints_num_keep', type=int, default=10, help='Number of checkpoints to keep')
    parser.add_argument('--save_dir', default='./output_standalone', type=str, help='Where to save training outputs')

    # Mindspore
    parser.add_argument('--amp_level', default='O3', help='Level for mixed precision training')
    parser.add_argument('--sink_mode', action='store_true', default=True, help='dataset sink mode')
    parser.add_argument('--distributed', type=str, default=False, help='distribute train')
    parser.add_argument('--device_num', type=int, default=8, help='ID of the target device')
    parser.add_argument('--device_target', default='Ascend', help='Device')
    parser.add_argument('--device_id', default=2, type=int, help='ID of the target device')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--lr_de_epochs', default=2, type=int)
    parser.add_argument('--context', default='gr', type=str)
    parser.add_argument('--is_fp16', type=str, default=True, help='cast to fp16 or not')

    config = parser.parse_args()

    main(config)
