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
"""export"""
import os
from pathlib import Path

from mindspore import context
from mindspore import dtype as mstype
from mindspore import export
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import numpy as mnp

from model_utils.config import config
from src.detr import build_detr
from src.utils import check_args


def run_export():
    """run export"""
    check_args(config)
    ckpt_path = Path(config.ckpt_path)
    file_format = config.file_format
    output_file_name = ckpt_path.parent / ckpt_path.stem

    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.device_target,
        device_id=device_id
    )

    detr = build_detr(config)
    load_param_into_net(detr, load_checkpoint(ckpt_path.as_posix()))

    image = mnp.zeros((1, 3, config.max_img_size, config.max_img_size), dtype=mstype.float32)
    mask = mnp.zeros((1, config.max_img_size, config.max_img_size), dtype=mstype.int32)

    print('Start export process')
    export(detr, image, mask, file_name=output_file_name.as_posix(), file_format=file_format)
    print(f'{output_file_name}.mindir exported successfully!')


if __name__ == "__main__":
    run_export()
