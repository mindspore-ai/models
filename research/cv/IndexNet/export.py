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
"""Export to MINDIR."""
from pathlib import Path

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore.train.serialization import export

from src.cfg.config import config as default_config
from src.model import MobileNetV2UNetDecoderIndexLearning


def _calculate_size(size, stride, odd):
    new_size = np.ceil(size / stride) * stride
    if odd:
        new_size += 1

    return int(new_size)


def run_export(config):
    """
    Export model to MINDIR.

    Args:
        config: Config parameters.
    """
    model = MobileNetV2UNetDecoderIndexLearning(
        encoder_rate=config.rate,
        encoder_current_stride=config.current_stride,
        encoder_settings=config.inverted_residual_setting,
        output_stride=config.output_stride,
        width_mult=config.width_mult,
        conv_operator=config.conv_operator,
        decoder_kernel_size=config.decoder_kernel_size,
        apply_aspp=config.apply_aspp,
        use_nonlinear=config.use_nonlinear,
        use_context=config.use_context,
    )

    load_checkpoint(config.ckpt_url, model)
    model.set_train(False)

    # Correctly process input
    odd_input = config.input_size % 2 == 1
    h = _calculate_size(config.img_size[0], config.output_stride, odd_input)
    w = _calculate_size(config.img_size[1], config.output_stride, odd_input)
    model_input = Tensor(np.ones([1, 4, h, w]), mstype.float32)

    save_path = Path(config.ckpt_url).resolve().with_suffix('').as_posix()

    export(model, model_input, file_name=save_path, file_format='MINDIR')
    print('Model exported successfully!')
    print(f'Path to exported model {save_path}.mindir')


if __name__ == "__main__":
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=default_config.device_target,
        device_id=default_config.device_id,
    )

    run_export(default_config)
