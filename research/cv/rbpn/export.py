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
import mindspore
from mindspore import Tensor, export, load_checkpoint, load_param_into_net, context
from src.model.rbpn import Net as RBPN




parser = argparse.ArgumentParser(description='rbpn export')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--file_name", type=str, default="rbpn_cloud", help="output file name.")
parser.add_argument("--file_format", type=str, default="MINDIR", choices=['MINDIR', 'AIR', 'ONNX'], help="file format")
parser.add_argument('--scale', type=int, default='4', help='super resolution scale')
parser.add_argument("--device_id", type=int, default=3, help="device id, default: 0.")
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument("--ckpt", type=str, default=r'./weights/rbpn.ckpt')
args = parser.parse_args()
print(args)
mindspore.set_seed(123)

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id)

    print("=======> load model ckpt")
    params = load_checkpoint(args.ckpt)
    model = RBPN(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=5, nFrames=7,
                 scale_factor=4)
    load_param_into_net(model, params)
    model.set_train(False)

    input_array = Tensor(np.zeros([1, 3, 120, 180], np.float32))
    neighbor_array = Tensor(np.zeros([1, 6, 3, 120, 180], np.float32))
    flow_array = Tensor(np.zeros([1, 6, 2, 120, 180], np.float32))

    input_all = [input_array, neighbor_array, flow_array]

    G_file = "{}_model".format(args.file_name)
    mindir_path = 'mindir_path'
    file_path = osp.join(os.getcwd(), mindir_path)
    if not osp.exists(file_path):
        os.makedirs(file_path)
    G_file_path = os.path.join(file_path, G_file)
    export(model, *input_all, file_name=G_file_path, file_format=args.file_format)
    print('export successfully!')
