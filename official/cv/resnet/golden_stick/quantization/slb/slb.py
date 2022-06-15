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
"""Create SLB-QAT algorithm instance."""

from mindspore_gs.quantization.slb import SlbQuantAwareTraining as QBNNQAT
from mindspore_gs.quantization.constant import QuantDtype

def create_slb(quant_type="graph_W4A8"):
    algo = QBNNQAT()
    if "W4A8" in quant_type:
        algo.set_weight_quant_dtype(QuantDtype.INT4)
        algo.set_act_quant_dtype(QuantDtype.INT8)
    elif "W2A8" in quant_type:
        algo.set_weight_quant_dtype(QuantDtype.INT2)
        algo.set_act_quant_dtype(QuantDtype.INT8)
    elif "W1A8" in quant_type:
        algo.set_weight_quant_dtype(QuantDtype.INT1)
        algo.set_act_quant_dtype(QuantDtype.INT8)
    elif "W1A4" in quant_type:
        algo.set_weight_quant_dtype(QuantDtype.INT1)
        algo.set_act_quant_dtype(QuantDtype.INT4)
    return algo
