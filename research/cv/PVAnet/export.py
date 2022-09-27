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

import os
import sys
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.PVANet.pva_faster_rcnn import PVANet_Infer
from src.model_utils.config import config
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")


if __name__ == '__main__':

    device_target = 'Ascend'
    device_id = 1

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    if device_target in ["Ascend", "GPU"]:
        context.set_context(device_id=device_id)

    net = PVANet_Infer(config)
    checkpoints = load_checkpoint(config.checkpoint_path)
    for oldkey in list(checkpoints.keys()):
        if not oldkey.startswith(('network',)):
            data = checkpoints.pop(oldkey)
            newkey = 'network.' + oldkey
            checkpoints[newkey] = data
            oldkey = newkey
    load_param_into_net(net, checkpoints, strict_load=True)

    # net.to_float(ms.float32)
    if device_target == "Ascend":
        net.to_float(ms.float16)
        print('cast to float16')

    img = Tensor(np.zeros([config.test_batch_size, 3, config.img_height, config.img_width]), ms.float32)

    img_metas = Tensor(np.random.uniform(0.0, 1.0, size=[config.test_batch_size, 4]), ms.float32)

    # file_format choose in ["AIR", "MINDIR"]
    export(net, img, img_metas, file_name=config.file_name, file_format=config.file_format)

    print('convert over')
