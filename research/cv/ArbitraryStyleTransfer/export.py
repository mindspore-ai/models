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
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import export
from src.model import style_transfer_model

parser = argparse.ArgumentParser(description="style transfer export")
parser.add_argument('--file_name', type=str, default='style_transfer', help='output file name prefix.')
parser.add_argument('--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'], default='MINDIR', \
                    help='file format')
parser.add_argument("--ckpt_path", type=str, default='./ckpt/style_transfer_model_0100.ckpt')
parser.add_argument('--platform', type=str, default='GPU', help='only support GPU')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")

if __name__ == '__main__':
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, device_id=args.device_id)
    model = style_transfer_model(style_dim=100)
    params = load_checkpoint(args.ckpt_path)
    load_param_into_net(model, params)
    model.set_train(True)
    input1_shape = [1, 3, 256, 256]
    input2_shape = [1, 768, 15, 15]
    input_array_1 = Tensor(np.random.uniform(-1.0, 1.0, size=input1_shape).astype(np.float32))
    input_array_2 = Tensor(np.random.uniform(-1.0, 1.0, size=input2_shape).astype(np.float32))
    G_file = f"{args.file_name}_model"
    export(model, input_array_1, input_array_2, file_name=G_file, file_format=args.file_format)
