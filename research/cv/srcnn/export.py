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
"""srcnn training"""

import numpy as np
from mindspore import Tensor, export
from mindspore.train.serialization import load_checkpoint

from src.srcnn import SRCNN

from src.model_utils.config import config

def run_export():
    cfg = config
    srcnn = SRCNN()
    # load the parameter into net
    load_checkpoint(cfg.checkpoint_path, net=srcnn)
    input_size = np.random.uniform(0.0, 1.0, size=[1, 1, cfg.image_width, cfg.image_height]).astype(np.float32)
    export(srcnn, Tensor(input_size), file_name='srcnn', file_format=cfg.file_format)

if __name__ == '__main__':
    run_export()
