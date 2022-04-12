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
MIMO-UNet export mindir.
"""
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import export
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from src.config import config
from src.mimo_unet import MIMOUNet


def run_export(args):
    """run export"""
    context.set_context(mode=context.GRAPH_MODE, device_target=args.export_device_target)
    context.set_context(device_id=args.device_id)

    net = MIMOUNet()

    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)
    input_shp = [args.export_batch_size, 3, 256, 256]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))

    export(net, input_array, file_name=args.export_file_name, file_format=args.export_file_format)


if __name__ == '__main__':
    run_export(config)
