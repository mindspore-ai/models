# Copyright 2021 Huawei Technologies Co., Ltd
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
from argparse import ArgumentParser
import numpy as np
from mindspore import Tensor, context, load_checkpoint, export
from src.model import ERFNet

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--device_target', default="Ascend", type=str)
    config = parser.parse_args()

    net = ERFNet(1, 20, "XavierUniform", run_distribute=False, train=False)

    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target=config.device_target)
    context.set_context(device_id=0)

    load_checkpoint(config.model_path, net=net)
    net.set_train(False)
    input_data = Tensor(np.zeros([1, 3, 512, 1024]).astype(np.float32))
    export(net, input_data, file_name="ERFNet.mindir", file_format="MINDIR")
