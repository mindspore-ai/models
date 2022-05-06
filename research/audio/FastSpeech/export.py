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
"""Run export"""
from pathlib import Path

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore.train.serialization import export

from src.cfg.config import config as default_config
from src.model import FastSpeech


def run_export(config):
    """
    Export model to MINDIR.
    """
    model = FastSpeech()

    load_checkpoint(config.fs_ckpt_url, model)
    model.set_train(False)

    input_1 = Tensor(np.ones([1, config.character_max_length]), dtype=mstype.float32)
    input_2 = Tensor(np.ones([1, config.character_max_length]), dtype=mstype.float32)
    name = Path(config.fs_ckpt_url).stem
    path = Path(config.fs_ckpt_url).resolve().parent
    save_path = str(Path(path, name))

    export(model, input_1, input_2, file_name=save_path, file_format='MINDIR')
    print('Model exported successfully!')
    print(f'Path to exported model {save_path}.mindir')


if __name__ == "__main__":
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=default_config.device_target,
        device_id=default_config.device_id,
    )

    run_export(default_config)
