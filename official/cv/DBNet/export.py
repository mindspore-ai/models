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
"""DBNet MindIR export."""
import os
import sys

import mindspore as ms

from src.utils.env import init_env
from src.modules.model import get_dbnet
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@moxing_wrapper()
def export():
    config.device_num = 1
    init_env(config)
    config.backbone.pretrained = False
    eval_net = get_dbnet(config.net, config, isTrain=False)
    ms.load_checkpoint(config.ckpt_path, eval_net)
    eval_net.set_train(False)
    inp = ms.ops.ones((1, 3, *config.eval.eval_size), ms.float32)
    file_name = os.path.join(config.output_dir,
                             config.net + '_' + config.backbone.initializer)
    ms.export(eval_net, inp, file_name=file_name, file_format='MINDIR')
    print("MINDIR saved at", file_name+".mindir")


if __name__ == '__main__':
    export()
