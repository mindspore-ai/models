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
"""export checkpoint file into air, onnx, mindir models"""

import argparse

import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.config import CONFIG
from src.ctrl import CTRL
cfg = CONFIG(data_dir='./')
parser = argparse.ArgumentParser(description='TALL export')
parser.add_argument("--device_id", type=int, default=0, help="device id")
parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint file path.")
parser.add_argument("--file_name", type=str, default="TALL", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend"], default="Ascend", help="device target")

args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    param_dict = load_checkpoint(args.checkpoint_path)
    net = CTRL(cfg.visual_dim, cfg.sentence_embed_dim, cfg.semantic_dim, cfg.middle_layer_dim)
    net.set_train(False)
    load_param_into_net(net, param_dict)
    batch_size = cfg.test_batch_size
    data = np.random.uniform(0.0, 1.0, size=[batch_size, cfg.visual_dim + cfg.sentence_embed_dim]).astype(np.float32)

    export(net, Tensor(data), file_name=args.file_name, file_format=args.file_format)
