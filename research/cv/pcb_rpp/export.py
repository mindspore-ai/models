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

""" export checkpoint file into mindir models"""

import os

import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from mindspore import dtype as mstype

from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper
from src.pcb import PCBInfer
from src.rpp import RPPInfer


def build_model():
    """ Create network """
    model = None
    if config.model_name == "PCB":
        model = PCBInfer()
    elif config.model_name == "RPP":
        model = RPPInfer()
    return model


def modelarts_pre_process():
    """ modelarts pre process function """
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=get_device_id())

    # define network
    network = build_model()

    assert config.checkpoint_file_path is not None, "checkpoint_path is None."

    # load network checkpoint
    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(network, param_dict)

    # export network
    inputs = Tensor(np.zeros([config.batch_size, 3, config.image_height, config.image_width]), mstype.float32)
    export(network, inputs, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    run_export()
