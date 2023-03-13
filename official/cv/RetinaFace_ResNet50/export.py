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
"""EXPORT ONNX MODEL WITH CKPT MODEL BASED ON MINDSPORE"""
from __future__ import print_function
import numpy as np
import mindspore as ms
from mindspore import Tensor, export
from src.network import RetinaFace, resnet50
from src.config import cfg_res50


def export_ONNX_model():
    cfg = cfg_res50

    ms.set_context(mode=ms.GRAPH_MODE, device_target=cfg.get('device'))

    # build network
    backbone = resnet50(1001)
    network = RetinaFace(phase='predict', backbone=backbone)
    backbone.set_train(False)
    network.set_train(False)

    # load checkpoint into network
    param_dict = ms.load_checkpoint(cfg['ckpt_model'])
    network.init_parameters_data()
    ms.load_param_into_net(network, param_dict)

    # build input data
    input_data = Tensor(np.ones([1, 3, 2176, 2176]).astype(np.float32))

    # export onnx model
    print("Begin to Export ONNX Model...")
    export(network, input_data, file_name='retinaface', file_format='ONNX')
    print("Export ONNX Model Successfully!")

if __name__ == '__main__':
    export_ONNX_model()
