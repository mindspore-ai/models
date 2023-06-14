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

import numpy as np
from mindspore import context, Tensor
from mindspore import load_checkpoint, export

from model_utils.device_adapter import get_device_id
from model_utils.config import config as cfg
from src.network import AutoEncoder

context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=get_device_id())


def get_network():
    auto_encoder = AutoEncoder(cfg)
    if cfg.model_arts:
        import moxing as mox

        mox.file.copy_parallel(src_url=cfg.checkpoint_url, dst_url=cfg.cache_ckpt_file)
        ckpt_path = cfg.cache_ckpt_file
    else:
        ckpt_path = cfg.checkpoint_path

    load_checkpoint(ckpt_path, net=auto_encoder)
    auto_encoder.set_train(False)
    return auto_encoder


def model_export():
    auto_encoder = get_network()
    channel = 1 if cfg.grayscale else 3
    input_size = cfg.crop_size
    batch_size = ((cfg.mask_size - cfg.crop_size) // cfg.stride + 1) ** 2
    input_data = Tensor(np.ones([batch_size, channel, input_size, input_size], np.float32))
    export(auto_encoder, input_data, file_name=f"SSIM-AE-{cfg.dataset}", file_format="MINDIR")


if __name__ == "__main__":
    model_export()
