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
"""reading comprehension file"""
from mindspore.common.initializer import Normal
from mindspore.common.tensor import Tensor
import mindspore.ops as ops
from mindspore.ops import composite as C
import mindspore.nn as nn
import mindspore

from src.luke.model import LukeEntityAwareAttentionModel


class LukeForReadingComprehensionWithLoss(nn.Cell):
    """read comprehension with loss"""

    def __init__(self, network):
        """init"""
        super(LukeForReadingComprehensionWithLoss, self).__init__()
        self.model1 = network
        self.loss_func = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.squeeze = ops.Squeeze(axis=1)
        self.clamp = C.clip_by_value
        self.min_value = Tensor(0, mindspore.int32)
        self.max_value = Tensor(512, mindspore.int32)

    def construct(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                  entity_segment_ids, entity_attention_mask, start_positions, end_positions):
        """construct fun"""
        start_logits, end_logits = self.model1(word_ids, word_segment_ids, word_attention_mask,
                                               entity_ids, entity_position_ids,
                                               entity_segment_ids, entity_attention_mask)
        start_positions = self.squeeze(start_positions)
        end_positions = self.squeeze(end_positions)
        start_positions = self.clamp(start_positions, self.min_value, self.max_value)
        end_positions = self.clamp(end_positions, self.min_value, self.max_value)
        start_loss = self.loss_func(start_logits, start_positions)
        end_loss = self.loss_func(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2.0
        return total_loss


class LukeForReadingComprehension(nn.Cell):
    """luke reading comprehension"""

    def __init__(self, config):
        """init"""
        super(LukeForReadingComprehension, self).__init__()
        self.luke = LukeEntityAwareAttentionModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, 2, weight_init=Normal(sigma=0.02))
        self.split = ops.Split(-1, 2)
        self.squeeze = ops.Squeeze(-1)
        self.cast = ops.Cast()

    def construct(self, word_ids, word_segment_ids, word_attention_mask, entity_ids=None, entity_position_ids=None,
                  entity_segment_ids=None, entity_attention_mask=None):
        """construct fun"""
        encoder_outputs = self.luke(word_ids, word_segment_ids, word_attention_mask,
                                    entity_ids, entity_position_ids,
                                    entity_segment_ids, entity_attention_mask)

        word_hidden_states = encoder_outputs[0][:, : ops.shape(word_ids)[1], :]
        word_hidden_states = self.cast(word_hidden_states, mindspore.float32)
        logits = self.qa_outputs(word_hidden_states)
        start_logits, end_logits = self.split(logits)
        start_logits = self.squeeze(start_logits)
        end_logits = self.squeeze(end_logits)
        return start_logits, end_logits
