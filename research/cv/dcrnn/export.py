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
import numpy as np
import mindspore
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.utils import load_graph_data
from src.dcrnn_model import DCRNNModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='src/dcrnn-1_375.ckpt', type=str, help='path to pretrained model')
    parser.add_argument('--file_name', default='DCRNN_mindir', type=str, help='export file name')
    parser.add_argument('--file_format', default='MINDIR', type=str, help='export file format')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size, Preferably the same as during training')
    parser.add_argument('--device', default='Ascend', help='Device string')
    parser.add_argument('--device_id', default=0, type=int, help='ID of the target device')
    parser.add_argument('--config_filename',
                        default='data/model/dcrnn_la.yaml', type=str,
                        help='Configuration filename for restoring the model')

    config = parser.parse_args()
    return config


def run_export():
    config = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device, device_id=config.device_id)

    with open(config.config_filename) as f:
        print('Start Reading Config File')
        supervisor_config = yaml.safe_load(f)
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        _, _, adj_mx = load_graph_data(graph_pkl_filename)
        _model_kwargs = supervisor_config.get('model')

    dcrnn_model = DCRNNModel(adj_mx, False, supervisor_config.get('data').get('batch_size'), **_model_kwargs)

    param_dict = load_checkpoint(config.checkpoint_path)
    param_not_load, _ = load_param_into_net(dcrnn_model, param_dict)
    print('param_not_load:', param_not_load)
    dcrnn_model.set_train(False)

    input_data = mindspore.Tensor(
        np.ones([config.batch_size, 2, 12, 207, 2]),
        mindspore.float32)
    print('Start export')
    export(dcrnn_model, input_data, file_name=config.file_name, file_format=config.file_format)
    print('Finish export')


if __name__ == '__main__':
    run_export()
