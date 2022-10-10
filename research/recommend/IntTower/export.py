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
import mindspore as ms
from model import IntTower

if __name__ == '__main__':
    network = IntTower()
    input_tensor = ms.Tensor(np.ones([2048, 7]).astype(np.float32))
    param_dict = ms.load_checkpoint("./IntTower.ckpt")
    ms.load_param_into_net(network, param_dict)
    ms.export(net=network, inputs=input_tensor, file_name='./IntTower', file_format="MINDIR")
