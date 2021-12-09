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
"""convert apdrawinggan auxiliary from pth to ckpt"""

import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

param = {
    # pylint: disable=C0301
    "myTrainOneStepCellForG.network.netDT1.model.model.0.weight": "myTrainOneStepCellForG.network.netDT1.model.model.0.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.1.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.1.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.2.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.2.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.2.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.5.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.5.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.6.weight": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.6.bias": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.6.beta",
    "myTrainOneStepCellForG.network.netDT1.model.model.3.weight": "myTrainOneStepCellForG.network.netDT1.model.model.3.weight",
    "myTrainOneStepCellForG.network.netDT1.model.model.3.bias": "myTrainOneStepCellForG.network.netDT1.model.model.3.bias",
    "myTrainOneStepCellForG.network.netDT2.model.model.0.weight": "myTrainOneStepCellForG.network.netDT2.model.model.0.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.1.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.1.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.2.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.2.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.2.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.5.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.6.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.6.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.5.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.5.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.6.weight": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.6.gamma",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.6.bias": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.6.beta",
    "myTrainOneStepCellForG.network.netDT2.model.model.3.weight": "myTrainOneStepCellForG.network.netDT2.model.model.3.weight",
    "myTrainOneStepCellForG.network.netDT2.model.model.3.bias": "myTrainOneStepCellForG.network.netDT2.model.model.3.bias",
    "myTrainOneStepCellForG.network.netLine1.model.model.0.weight": "myTrainOneStepCellForG.network.netLine1.model.model.0.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.1.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.1.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.2.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.2.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.2.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.5.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.5.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.6.weight": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.6.bias": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.6.beta",
    "myTrainOneStepCellForG.network.netLine1.model.model.3.weight": "myTrainOneStepCellForG.network.netLine1.model.model.3.weight",
    "myTrainOneStepCellForG.network.netLine1.model.model.3.bias": "myTrainOneStepCellForG.network.netLine1.model.model.3.bias",
    "myTrainOneStepCellForG.network.netLine2.model.model.0.weight": "myTrainOneStepCellForG.network.netLine2.model.model.0.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.1.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.1.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.2.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.2.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.2.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.1.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.5.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.5.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.6.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.6.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.6.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.5.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.5.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.6.weight": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.6.gamma",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.6.bias": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.6.beta",
    "myTrainOneStepCellForG.network.netLine2.model.model.3.weight": "myTrainOneStepCellForG.network.netLine2.model.model.3.weight",
    "myTrainOneStepCellForG.network.netLine2.model.model.3.bias": "myTrainOneStepCellForG.network.netLine2.model.model.3.bias",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT1.model.model.1.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT1.model.model.1.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.6.moving_mean": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netDT2.model.model.1.model.6.moving_variance": "myTrainOneStepCellForG.network.netDT2.model.model.1.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine1.model.model.1.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine1.model.model.1.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.2.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.model.4.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.3.model.6.moving_variance",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.6.moving_mean": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.6.moving_mean",
    "myTrainOneStepCellForG.network.netLine2.model.model.1.model.6.moving_variance": "myTrainOneStepCellForG.network.netLine2.model.model.1.model.6.moving_variance"
}



def pytorch2mindspore():
    """pytorch2mindspore"""
    DT1_par_dict = torch.load('auxiliary/latest_net_DT1.pth', map_location='cpu')
    DT2_par_dict = torch.load('auxiliary/latest_net_DT2.pth', map_location='cpu')
    Line1_par_dict = torch.load('auxiliary/latest_net_Line1.pth', map_location='cpu')
    Line2_par_dict = torch.load('auxiliary/latest_net_Line2.pth', map_location='cpu')
    new_params_list = []
    for name in DT1_par_dict:
        param_dict = {}
        tran_name = name
        if 'num_batches_tracked' in tran_name:
            continue
        parameter = DT1_par_dict[name]
        tran_name = tran_name.replace('running_mean', 'moving_mean')
        tran_name = tran_name.replace('running_var', 'moving_variance')
        tran_name = 'myTrainOneStepCellForG.network.netDT1.' + tran_name
        ms_name = param[tran_name]
        param_dict['name'] = ms_name
        param_dict['data'] = Tensor(parameter.detach().numpy())
        new_params_list.append(param_dict)

    for name in DT2_par_dict:
        param_dict = {}
        tran_name = name
        if 'num_batches_tracked' in tran_name:
            continue
        parameter = DT2_par_dict[name]
        tran_name = tran_name.replace('running_mean', 'moving_mean')
        tran_name = tran_name.replace('running_var', 'moving_variance')
        tran_name = 'myTrainOneStepCellForG.network.netDT2.' + tran_name
        ms_name = param[tran_name]
        param_dict['name'] = ms_name
        param_dict['data'] = Tensor(parameter.detach().numpy())
        new_params_list.append(param_dict)

    for name in Line1_par_dict:
        param_dict = {}
        tran_name = name
        if 'num_batches_tracked' in tran_name:
            continue
        parameter = Line1_par_dict[name]
        tran_name = tran_name.replace('running_mean', 'moving_mean')
        tran_name = tran_name.replace('running_var', 'moving_variance')
        tran_name = 'myTrainOneStepCellForG.network.netLine1.' + tran_name
        ms_name = param[tran_name]
        param_dict['name'] = ms_name
        param_dict['data'] = Tensor(parameter.detach().numpy())
        new_params_list.append(param_dict)

    for name in Line2_par_dict:
        param_dict = {}
        tran_name = name
        if 'num_batches_tracked' in tran_name:
            continue
        parameter = Line2_par_dict[name]
        tran_name = tran_name.replace('running_mean', 'moving_mean')
        tran_name = tran_name.replace('running_var', 'moving_variance')
        tran_name = 'myTrainOneStepCellForG.network.netLine2.' + tran_name
        ms_name = param[tran_name]
        param_dict['name'] = ms_name
        param_dict['data'] = Tensor(parameter.detach().numpy())
        new_params_list.append(param_dict)
    save_checkpoint(new_params_list, 'auxiliary.ckpt')

pytorch2mindspore()
