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
from mindspore.train.serialization import export, load_param_into_net, load_checkpoint

from src.delf_model import Model as DELF

parser = argparse.ArgumentParser(description='Export MINDIR')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument('--ckpt_path', type=str, default='')

args = parser.parse_known_args()[0]

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)

if __name__ == '__main__':

    delf_net = DELF(state="test")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(delf_net, param_dict)

    input_batch = Tensor(np.random.uniform(
        -1.0, 1.0, size=(7, 3, 2048, 2048)).astype(np.float32))

    export(delf_net, input_batch, file_name='DELF_MindIR', file_format='MINDIR')
    print("Export successfully!")
