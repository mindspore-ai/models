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
"""Export checkpoint file into MINDIR format"""

from pathlib import Path

import mindspore.common.dtype as mstype
import mindspore.numpy as mnp
from mindspore import context
from mindspore import export
from mindspore import load_checkpoint

from model_utils.config import get_config
from src.generator.generator import Generator
from src.utils import check_args


def export_ctsdg(cfg):
    """
    Export CTSDG generator for inference

    Args:
        cfg: Model configuration

    Returns:
        None
    """
    generator = Generator(
        image_in_channels=config.image_in_channels,
        edge_in_channels=config.edge_in_channels,
        out_channels=config.out_channels
    )
    generator.set_train(False)
    load_checkpoint(cfg.checkpoint_path, generator)

    ckpt_path = Path(cfg.checkpoint_path)
    output_file_name = (ckpt_path.parent / ckpt_path.stem).as_posix()
    file_format = config.file_format

    img_dummy = mnp.zeros([1, config.image_in_channels, *cfg.image_load_size],
                          dtype=mstype.float32)
    edge_dummy = mnp.zeros([1, 2, *cfg.image_load_size], dtype=mstype.float32)
    mask_dummy = mnp.zeros([1, 1, *cfg.image_load_size], dtype=mstype.float32)

    export(generator, img_dummy, edge_dummy, mask_dummy,
           file_name=output_file_name, file_format=file_format)

    print(f'{output_file_name}.mindir exported successfully!', flush=True)


if __name__ == '__main__':
    config = get_config()
    check_args(config)
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.device_target,
        device_id=config.device_id,
    )
    export_ctsdg(config)
