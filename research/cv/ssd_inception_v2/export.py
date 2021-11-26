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
"""Export checkpoint file into mindir format"""

import os

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.train.serialization import export
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from src.box_utils import default_boxes
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.ssd import ssd_inception_v2, SsdInferWithDecoder

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


def modelarts_pre_process():
    """
    Modelarts pre process function

    Returns:

    """
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """
    Run export

    Returns:

    """
    if hasattr(config, 'num_ssd_boxes') and config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.steps[i]) * (w // config.steps[i]) * config.num_default[i]
        config.num_ssd_boxes = num

    if config.model_name == "ssd_inception_v2":
        net = ssd_inception_v2(config=config)
    else:
        raise ValueError(f'config.model: {config.model_name} is not supported')

    net = SsdInferWithDecoder(net, Tensor(default_boxes), config)

    param_dict = load_checkpoint(config.checkpoint_file_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_shp = [config.batch_size, 3] + config.img_shape
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp), mstype.float32)
    export(net, input_array, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    run_export()
