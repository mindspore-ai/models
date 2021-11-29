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
import argparse
import numpy as np

from src.options.general import opts
from src.models.ADNet import adnet

from mindspore import Tensor, export, context

parser = argparse.ArgumentParser(
    description='ADNet test')
parser.add_argument('--weight_file', default='', type=str, help='The pretrained weight file')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument('--target_device', type=int, default=0)
args = parser.parse_args()
context.set_context(device_target=args.device_target, mode=context.PYNATIVE_MODE, device_id=args.target_device)
opts['num_videos'] = 1
net, domain_specific_nets = adnet(opts, trained_file=args.weight_file)

input_ = np.random.uniform(0.0, 1.0, size=[128, 3, 112, 112]).astype(np.float32)
export(net, Tensor(input_), file_name='ADNet', file_format='MINDIR')
print('export finished')
