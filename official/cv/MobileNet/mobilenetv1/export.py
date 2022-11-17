# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore as ms

from src.mobilenet_v1 import mobilenet_v1 as mobilenet
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_export_preprocess


ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)


@moxing_wrapper(pre_process=modelarts_export_preprocess)
def export_mobilenetv1():
    """ export_mobilenetv1 """
    target = config.device_target
    if target != "GPU":
        ms.set_context(device_id=get_device_id())

    network = mobilenet(class_num=config.class_num)
    ms.load_checkpoint(config.ckpt_file, net=network)
    network.set_train(False)
    input_data = ms.numpy.zeros([config.batch_size, 3, config.height, config.width]).astype(np.float32)
    ms.export(network, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    export_mobilenetv1()
