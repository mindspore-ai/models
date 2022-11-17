# Copyright 2020 Huawei Technologies Co., Ltd
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
"""export file"""
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from src.efficientnet import efficientnet_b0
from src.config import config

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


if __name__ == "__main__":
    if config.device_target not in ("GPU", "CPU"):
        raise ValueError("Only supported CPU and GPU now.")

    net = efficientnet_b0(num_classes=config.num_classes,
                          drop_rate=config.drop,
                          drop_connect_rate=config.drop_connect,
                          global_pool=config.gp,
                          bn_tf=config.bn_tf,
                          )

    ckpt = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, ckpt)
    net.set_train(False)

    image = Tensor(np.ones([config.batch_size, 3, config.height, config.width], np.float32))
    export(net, image, file_name=config.file_name, file_format=config.file_format)
