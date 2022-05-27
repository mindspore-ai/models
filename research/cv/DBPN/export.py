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

"""export net together with checkpoint into air/mindir models"""
import os
import os.path as osp
import argparse
import numpy as np
from src.model.generator import get_generator
from mindspore import Tensor, context, export
from mindspore.train.serialization import load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser(description='dbpn export')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_path", type=str, required=True, help="path of checkpoint file")
parser.add_argument("--file_name", type=str, default="ddbpn", help="output file name.")
parser.add_argument("--file_format", type=str, default="MINDIR", choices=['MINDIR', 'AIR', 'ONNX'], help="file format")
parser.add_argument('--scale', type=int, default='4', help='super resolution scale')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument('--model_type', type=str, default='DDBPN', choices=["DBPNS", "DDBPN", "DBPN", "DDBPNL"])
args = parser.parse_args()

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id)
    model = args.model_type
    generator = get_generator(model, args.scale)
    params = load_checkpoint(args.ckpt_path)
    load_param_into_net(generator, params)
    generator.set_train(False)
    print('load mindspore net and checkpoint successfully.')
    input_shp = [1, 3, 200, 200]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    G_file = "{}_model".format(args.file_name)
    G_file_path = osp.join(os.getcwd(), G_file)
    if not osp.exists(G_file_path):
        os.makedirs(G_file_path)
    export(generator, input_array, file_name=G_file_path, file_format=args.file_format)
    print('export successfully!')
