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
"""Export models into different formats"""
import mindspore as ms
import mindspore.context as context
import numpy as np

from src.model.generator import HiFaceGANGenerator
from src.model_utils.config import get_config


def run_export(config):
    """Export HiFaceGAN generator network"""
    generator = HiFaceGANGenerator(
        ngf=config.ngf,
        input_nc=config.input_nc
    )
    ms.load_checkpoint(config.ckpt_file, net=generator)
    img = ms.Tensor(np.zeros([config.batch_size, config.input_nc, config.img_size, config.img_size]), ms.float32)
    ms.export(generator, img, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    cfg = get_config()
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    run_export(cfg)
