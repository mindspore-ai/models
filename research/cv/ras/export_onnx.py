"""
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

import argparse
import numpy as np
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net, export
from src.model import BoneModel


def run_export(device_target, device_id, pretrained_model, model_ckpt, batchsize):
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=device_target, device_id=device_id)
    net = BoneModel(device_target, pretrained_model)
    param_dict = load_checkpoint(model_ckpt)
    load_param_into_net(net, param_dict)
    input_arr = ms.Tensor(np.ones((batchsize, 3, 352, 352)).astype(np.float32))

    export(net, input_arr, file_name="ras_onnx", file_format='ONNX')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_target', type=str, default='GPU', help="device's name, Ascend,GPU,CPU")
    parser.add_argument('--device_id', type=int, default=5, help="Number of device")
    parser.add_argument('--batchsize', type=int, default=1, help="training batch size")
    parser.add_argument('--pre_model', type=str)
    parser.add_argument('--ckpt_file', type=str)
    par = parser.parse_args()


    run_export(par.device_target, int(par.device_id), par.pre_model, par.ckpt_file, par.batchsize)
