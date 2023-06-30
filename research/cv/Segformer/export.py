# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import numpy as np

import mindspore as ms
from mindspore import Tensor, context, set_seed
from src.segformer import SegFormer
from src.model_utils.config import get_export_config


def run_export(config):
    assert config.export_ckpt_path is not None, "export_ckpt_path is None."
    assert config.export_format in ["AIR", "MINDIR", "ONNX"], "export_format should be in [AIR, MINDIR, ONNX]"

    net = SegFormer(config.backbone, config.class_num, sync_bn=config.run_distribute)
    param_dict = ms.load_checkpoint(config.export_ckpt_path)
    ms.load_param_into_net(net, param_dict)
    print(f"load ckpt from \"{config.export_ckpt_path}\" success.")

    # export
    h, w = config.base_size
    input_arr = Tensor(np.ones([1, 3, h, w]), ms.float32)
    file_name = os.path.basename(config.config_path)[:-5]
    ms.export(net, input_arr, file_name=file_name, file_format=config.export_format)
    print(f"export {config.export_format}: {file_name} success.")


if __name__ == '__main__':
    export_config = get_export_config()
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target=export_config.device_target)
    run_export(export_config)
