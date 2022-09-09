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
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from src.refinenet import RefineNet, Bottleneck

context.set_context(mode=context.GRAPH_MODE, save_graphs=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='checkpoint export')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint of refinenet (Default: None)')
    parser.add_argument('--num_classes', type=int, default=21, help='the number of classes (Default: 21)')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--crop_size', type=int, default=513, help='crop size')
    parser.add_argument('--file_name', type=str, default="refinenet", help='model name')
    parser.add_argument('--file_format', type=str, choices=["MINDIR", "AIR", "ONNX"], default="MINDIR",
                        help='model format')
    args = parser.parse_args()

    network = RefineNet(Bottleneck, [3, 4, 23, 3], args.num_classes)
    param_dict = load_checkpoint(args.checkpoint)

    # load the parameter into net
    load_param_into_net(network, param_dict)
    image_shape = [args.batch_size, 3, args.crop_size, args.crop_size]
    input_data = np.random.uniform(0.0, 1.0, size=image_shape).astype(np.float32)
    export(network, Tensor(input_data), file_name=args.file_name, file_format=args.file_format)
