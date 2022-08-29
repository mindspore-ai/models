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

from mindspore.nn import Cell
from src.model_utils.config import config
if config.device_target == 'GPU':
    from src.model.attn import RelPartialLearnableMultiHeadAttn
    from src.model.positionwiseFF import PositionwiseFF
else:
    from src.model.attn_for_ascend import RelPartialLearnableMultiHeadAttn
    from src.model.positionwiseFF_for_ascend import PositionwiseFF


class RelPartialLearnableDecoderLayer(Cell):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def construct(self, dec_inp, r, r_w_bias, r_r_bias, mems=None, attn_mask=None):
        output = self.attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output
