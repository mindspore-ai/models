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
"""
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.config import config

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

if __name__ == '__main__':
    if config.net == 'resnet50':
        from src.glore_resnet import glore_resnet50
        net = glore_resnet50(class_num=config.class_num)
    elif config.net == 'resnet200':
        from src.glore_resnet import glore_resnet200
        net = glore_resnet200(class_num=config.class_num)

    assert config.ckpt_url is not None, "config.ckpt_url is None."
    param_dict = load_checkpoint(config.ckpt_url)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.ones([config.batch_size, 3, 224, 224]), mstype.float32)
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)
