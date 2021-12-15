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
######################## export mindir ########################
export net as mindir
"""
import numpy as np
import mindspore
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.model import TCN
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())


@moxing_wrapper()
def export_tcn():
    """export net as mindir"""
    # define fusion network
    net = TCN(config.channel_size, config.num_classes, [config.nhid] * config.level, config.kernel_size, config.dropout
              , config.dataset_name)
    # load network checkpoint
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)

    # export network
    if config.dataset_name == 'permuted_mnist':
        inputs = Tensor(np.ones([config.batch_size, config.channel_size, config.image_height * config.image_width]),
                        mindspore.float32)
    elif config.dataset_name == 'adding_problem':
        inputs = Tensor(np.ones([config.batch_test, config.channel_size, config.seq_length]), mindspore.float32)
    export(net, inputs, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    export_tcn()
