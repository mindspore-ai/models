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
"""Export model to MINDIR"""

import os

import numpy as np
from mindspore import Tensor
from mindspore import export
from mindspore import load_checkpoint
from mindspore.common import dtype as mstype

from src.config import config
from src.vgg import vgg16


if __name__ == "__main__":
    path = config.ckpt_file
    net = vgg16(10, args=config)
    load_checkpoint(path, net=net)
    img = np.random.randint(0, 255, size=(1, 3, config.image_size, config.image_size))
    img = Tensor(np.array(img), mstype.float32)
    export(net, img, file_name=os.path.join(config.file_save, config.file_name), file_format=config.file_format)
