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
"""export midas."""
import numpy as np
from src.midas_net import MidasNet
from src.config import config
from mindspore import Tensor, export, context
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint

def midas_export():
    """export midas."""
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.device_target,
        save_graphs=False,
        device_id=config.device_id)
    net = MidasNet()
    load_checkpoint(config.model_weights, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([1, 3, config.img_width, config.img_height]), mstype.float32)
    if config.file_format == 'ONNX':
        if config.img_width == 160:
            export(net, input_data, file_name='midas_Sintel_1_3_160_384', file_format=config.file_format)
        if config.img_width == 384:
            export(net, input_data, file_name='midas_Kitti_1_3_384_1280', file_format=config.file_format)
        if config.img_width == 288:
            export(net, input_data, file_name='midas_TUM_1_3_288_384', file_format=config.file_format)
        else:
            export(net, input_data, file_name='midas-onnx', file_format=config.file_format)

    else:
        export(net, input_data, file_name='midas', file_format=config.file_format)

if __name__ == '__main__':
    midas_export()
