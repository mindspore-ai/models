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
"""
##############export checkpoint file into air and onnx models#################
python export.py
"""
import argparse
import numpy as np

from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from ISyNet.model import ISyNet

PARSER = argparse.ArgumentParser(description="IsyNet export")
PARSER.add_argument("--device_target", type=str, default='Ascend', help="device target")
PARSER.add_argument("--device_id", type=int, default=0, help="device id")
PARSER.add_argument("--jsonFile", type=str, default='./json/ISyNet-N0.json', help="json architecture description")
PARSER.add_argument("--weight_standardization", default=0, type=int, help="weight standardization")
PARSER.add_argument("--lastBN", type=int, default=1, help="last batch norm")
PARSER.add_argument("--batch_size", type=int, default=1, help="batch size")
PARSER.add_argument("--width", type=int, default=224, help="image width")
PARSER.add_argument("--height", type=int, default=224, help="image height")
PARSER.add_argument("--file_name", type=str, default='ISyNet-N0.json', help="output file name")
PARSER.add_argument("--file_format", type=str, default='MINDIR', help="output file format")
PARSER.add_argument("--checkpoint_file_path", type=str, default=None, help="checkpoint")
ARGS = PARSER.parse_args()


context.set_context(mode=context.GRAPH_MODE, device_target=ARGS.device_target)

if ARGS.device_target == "Ascend":
    context.set_context(device_id=ARGS.device_id)

def run_export():
    """run export."""

    net = ISyNet(num_classes=1001,
                 json_arch_file_backbone=ARGS.jsonFile,
                 dropout=0,
                 weight_standardization=ARGS.weight_standardization,
                 last_bn=ARGS.lastBN,
                 dml=0,
                 evaluate=True)

    assert ARGS.checkpoint_file_path is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(ARGS.checkpoint_file_path)

    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([ARGS.batch_size, 3, ARGS.height, ARGS.width], np.float32))
    export(net, input_arr, file_name=ARGS.file_name, file_format=ARGS.file_format)

if __name__ == '__main__':
    run_export()
