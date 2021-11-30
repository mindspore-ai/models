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
import os
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export
from src.models import get_r2plus1d_model
from src.config import config as cfg

if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target,
                        device_id=device_id)
    net = get_r2plus1d_model(cfg.num_classes, cfg.layer_num)

    param_dict = load_checkpoint(cfg.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([cfg.export_batch_size, 3, 16, \
                                cfg.image_height, cfg.image_width]), ms.float32)
    export(net, input_arr, file_name=cfg.file_name, file_format=cfg.file_format)
