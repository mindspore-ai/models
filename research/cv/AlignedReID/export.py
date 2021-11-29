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
""" export checkpoint file into models"""

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_param_into_net, load_checkpoint, export

from model_utils.config import get_config
from model_utils.moxing_adapter import moxing_wrapper
from src.aligned_reid import AlignedReID

config = get_config()


def modelarts_pre_process():
    """model arts pre process"""


@moxing_wrapper(pre_process=modelarts_pre_process)
def export_network():
    """ Export network """
    context.set_context(
        # mode=context.GRAPH_MODE,
        mode=context.PYNATIVE_MODE,
        device_target=config.device_target,
    )

    network = AlignedReID(num_classes=0)
    config.image_size = list(map(int, config.image_size.split(',')))

    print('Load model from', config.ckpt_file)
    ret = load_param_into_net(network, load_checkpoint(config.ckpt_file))
    print(ret)

    source_image = Tensor(np.ones((config.per_batch_size, 3, *config.image_size)).astype(np.float32))

    export(network, source_image, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    export_network()
