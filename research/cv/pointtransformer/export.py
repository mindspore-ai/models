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
"""
export file.
"""
import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net

from src.model.pointTransfomrer import create_cls_mode, create_seg_mode
from src.config.default import get_config
from src.utils.common import context_device_init



def export_net():
    config = get_config()
    context_device_init(config)
    if config.model_type == 'classification':
        network = create_cls_mode()
    elif config.model_type == 'segmentation':
        network = create_seg_mode()
    else:
        raise ValueError("Not support model type.")

    pretrain_ckpt_path = config.pretrain_ckpt
    load_param_into_net(network, load_checkpoint(pretrain_ckpt_path))
    input_shape = [config.batch_size, config.num_points, 6]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shape).astype(np.float32))
    export(network, input_array, file_name=config.file_name, file_format=config.file_format)

    print(f"Successful export {config.file_name}")

if __name__ == '__main__':
    export_net()
