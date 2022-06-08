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
# =======================================================================================
"""
##############export checkpoint file into air, mindir models#################
python export.py
"""
import os
import numpy as np

import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from model_utils.config import config
from src.yolox import DetectionBlock


def run_export():
    """
    Export the MINDIR file
    Returns:None

    """
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)
    if config.backbone == "yolox_darknet53":
        backbone = "yolofpn"
    else:
        backbone = "yolopafpn"
    network = DetectionBlock(config, backbone=backbone)  # default yolo-darknet53
    network.set_train(False)
    assert config.val_ckpt is not None, "config.ckpt_file is None."
    param_dict = load_checkpoint(config.val_ckpt)
    load_param_into_net(network, param_dict)
    input_arr = Tensor(np.ones([config.export_bs, 3, config.input_size[0], config.input_size[1]]), ms.float32)
    file_name = backbone
    export(network, input_arr, file_name=file_name, file_format=config.file_format)


if __name__ == '__main__':
    run_export()
