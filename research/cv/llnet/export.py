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
##############export checkpoint file into [AIR MINDIR ONNX] models#################
"""
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.llnet import  LLNet
from src.model_utils.config import config

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    if config.device_target == "Ascend" or config.device_target == "GPU":
        context.set_context(device_id=config.device_id)
    checkpoint = config.checkpoint
    if config.enable_modelarts:
        # download dataset from obs to server
        import moxing
        # download the checkpoint from obs to server
        if config.ckpt_url != '':
            print("=========================================================")
            print("config.ckpt_url  =", config.ckpt_url)
            base_name = os.path.basename(config.ckpt_url)
            dst_url = os.path.join(config.load_path, base_name)
            moxing.file.copy_parallel(src_url=config.ckpt_url, dst_url=dst_url)
            checkpoint = dst_url
            print("checkpoint =", checkpoint)

    net = LLNet()
    param_dict = load_checkpoint(checkpoint)
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.ones([1, 289]), ms.float32)
    export(net, input_data, file_name=config.file_name, file_format=config.file_format)

    if config.enable_modelarts:
        import moxing as mox
        print("=========================================================")
        print(os.listdir("."))
        src_url = './'
        print(src_url)
        if os.path.exists(src_url):
            print("config.result_url =", config.result_url)
            dst_url = config.result_url
            print("dst_url =", dst_url)
            mox.file.copy_parallel(src_url=src_url, dst_url=dst_url)
