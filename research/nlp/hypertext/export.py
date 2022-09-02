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
"""export checkpoint file into air models"""

import argparse
import numpy as np

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import ArgMaxWithValue
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import export

from src.config import Config
from src.hypertext import HModel

parser = argparse.ArgumentParser(description="hypertext export")
parser.add_argument('--modelPath', default='./output/hypertext_iflytek.ckpt', type=str, help='save model path')
parser.add_argument('--datasetType', default='iflytek', type=str, help='iflytek/tnews')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--file_name",
                    type=str,
                    default="hypertext",
                    help="output file name.")
parser.add_argument('--device', default='GPU', type=str, help='device GPU Ascend')
parser.add_argument('--file_format', default="MINDIR", type=str, choices=['AIR', "MINDIR"], help="file format")
args = parser.parse_args()

config = Config(None, None, args.device)
if args.datasetType == 'tnews':
    config.useTnews()
    config.n_vocab = 147919  # vocab size
    config.num_classes = 15  # label size
else:
    config.useIflyek()
    config.n_vocab = 118133  # vocab size
    config.num_classes = 119  # label size

if config.device == 'GPU':
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
elif config.device == 'Ascend':
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')


class HyperTextTextInferExportCell(Cell):
    """
    HyperText network infer.
    """

    def __init__(self, network):
        """init fun"""
        super(HyperTextTextInferExportCell, self).__init__(auto_prefix=False)
        self.network = network
        self.argmax = ArgMaxWithValue(axis=1, keep_dims=True)

    def construct(self, x1, x2):
        """construct hypertexttext infer cell"""
        predicted_idx = self.network(x1, x2)
        predicted_idx = self.argmax(predicted_idx)
        return predicted_idx


def run_export():
    hmodel = HModel(config)
    param_dict = load_checkpoint(args.modelPath)
    load_param_into_net(hmodel, param_dict)
    file_name = args.file_name + '_' + args.datasetType
    ht_infer = HyperTextTextInferExportCell(hmodel)
    x1 = Tensor(np.ones((args.batch_size, config.max_length)).astype(np.int32))
    x2 = Tensor(np.ones((args.batch_size, config.max_length)).astype(np.int32))
    export(ht_infer, x1, x2, file_name=file_name, file_format=args.file_format)


if __name__ == '__main__':
    run_export()
