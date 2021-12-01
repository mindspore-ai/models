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
from src.model.generator import Generator

parser = argparse.ArgumentParser(description="SRGAN export")
parser.add_argument('--file_name', type=str, default='SRGAN', help='output file name prefix.')
parser.add_argument('--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'], default='MINDIR', \
                    help='file format')
parser.add_argument("--generator_path", type=str, default='')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")

if __name__ == '__main__':
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id)
    generator = Generator(4)
    params = load_checkpoint(args.generator_path)
    load_param_into_net(generator, params)
    generator.set_train(False)
    input_shp = [1, 3, 126, 126]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    G_file = f"{args.file_name}_model"
    generator(input_array)
    export(generator, input_array, file_name=G_file, file_format=args.file_format)
