# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""
    export checkpoint file into air, onnx, mindir models
"""
import numpy as np
from mindspore import Tensor, context
from mindspore.train.serialization import export
from mindspore.train.serialization import load_checkpoint
from src.models.pix2pix import get_generator
from src.utils.config import config
from src.utils.moxing_adapter import moxing_wrapper

@moxing_wrapper()
def export_pix2pix():

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

    netG = get_generator()
    netG.set_train()
    print("CKPT:", config.ckpt)
    load_checkpoint(config.ckpt, netG)

    input_shp = [config.batch_size, 3, config.image_size, config.image_size]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    export(netG, input_array, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_pix2pix()
