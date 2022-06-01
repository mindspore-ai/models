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
##############export checkpoint file into air, onnx or mindir model#################
python export.py
"""

import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from mindspore import dtype as mstype

from src.args import args
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_model

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

if args.device_target in ["Ascend", "GPU"]:
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    net = get_model(args)
    criterion = get_criterion(args)
    cast_amp(net)
    net_with_loss = NetWithLoss(net, criterion)
    assert args.pretrained is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args.pretrained)
    load_param_into_net(net, param_dict)

    net.set_train(False)
    net.to_float(mstype.float32)

    input_arr = Tensor(np.zeros([1, 3, args.image_size, args.image_size], np.float32))
    export(net, input_arr, file_name=args.arch, file_format=args.file_format)
