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
"""export file."""
import argparse
import numpy as np

from mindspore import context, Tensor, load_checkpoint
from mindspore.train.serialization import export, load_param_into_net
from src.utils import get_network


parser = argparse.ArgumentParser(description='DeepID_export')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--input_dim', type=int, default=3, help='image dim')
parser.add_argument('--num_class', type=int, default=1283, help='number of classes')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument("--ckpt_path", type=str, default='./ckpt_path_11/', help="Checkpoint saving path.")
parser.add_argument("--run_distribute", type=int, default=0, help="Run distribute, default: 0.")
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument("--device_target", type=str, default='Ascend', help="device target")
parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")
parser.add_argument("--file_format", type=str, default='MINDIR', choices=["AIR", "ONNX", "MINDIR"],
                    help='file format')

if __name__ == '__main__':

    args_opt = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    deepid = get_network(args_opt, args_opt.num_class)
    param_network = load_checkpoint(args_opt.ckpt_path)
    load_param_into_net(deepid, param_network)

    deepid.set_train(False)

    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=(1, 3, 55, 47)).astype(np.float32))
    output_file = f"DeepID"
    export(deepid, input_array, file_name=output_file, file_format=args_opt.file_format)
