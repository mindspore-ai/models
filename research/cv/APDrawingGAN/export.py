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
##############export checkpoint file into onnx models#################
python export.py
"""
import numpy as np

import mindspore as ms
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.models.APDrawingGAN_G import Generator
from src.option.options_test import TestOptions
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

if __name__ == '__main__':
    print(ms.__version__)
    opt = TestOptions().get_settings()
    opt.isExport = True
    net = Generator(opt)
    param_dict = load_checkpoint(opt.model_path)
    load_param_into_net(net, param_dict)

    real_A = Tensor(np.ones([1, 3, 512, 512]) * 1.0, ms.float32)
    real_A_bg = Tensor(np.ones([1, 3, 512, 512]) * 1.0, ms.float32)
    real_A_eyel = Tensor(np.ones([1, 3, 80, 112]) * 1.0, ms.float32)
    real_A_eyer = Tensor(np.ones([1, 3, 80, 112]) * 1.0, ms.float32)
    real_A_nose = Tensor(np.ones([1, 3, 96, 96]) * 1.0, ms.float32)
    real_A_mouth = Tensor(np.ones([1, 3, 80, 128]) * 1.0, ms.float32)
    real_A_hair = Tensor(np.ones([1, 3, 512, 512]) * 1.0, ms.float32)
    mask = Tensor(np.ones([1, 1, 512, 512]) * 1.0, ms.float32)
    mask2 = Tensor(np.ones([1, 1, 512, 512]) * 1.0, ms.float32)
    center = np.array([[199., 238.],
                       [313., 238.],
                       [254., 300.],
                       [256., 369.]])
    net.set_pad(center)

    input_arr = [real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose, real_A_mouth, real_A_hair, mask, mask2]
    export(net, *input_arr, file_name=opt.mindir_filename, file_format="MINDIR")
    if opt.isModelarts:
        from src.utils.tools import modelarts_result2obs
        modelarts_result2obs(opt)
