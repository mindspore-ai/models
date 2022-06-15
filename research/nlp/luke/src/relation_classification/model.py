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
RelationClassification model
"""
from mindspore import nn
from mindspore import ops
from mindspore.nn import SoftmaxCrossEntropyWithLogits

from src.luke.tacred_model import LukeEntityAwareAttentionModel


class LukeForRelationClassification(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super().__init__(args)
        self.loss_func = SoftmaxCrossEntropyWithLogits(sparse=True)

        self.args = args

        self.num_labels = num_labels
        self.dropout = nn.Dropout(1 - args.hidden_dropout_prob)
        self.classifier = nn.Dense(args.hidden_size * 2, num_labels, has_bias=False)
        self.op = ops.Concat(axis=1)

    def construct(
            self,
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
            label=None,
    ):
        encoder_outputs = super().construct(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )
        feature_vector = self.op((encoder_outputs[1][:, 0, :], encoder_outputs[1][:, 1, :]))
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)
        return self.loss_func(logits, label[:, 0])


class LukeForRelationClassificationEval(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels):
        super().__init__(args)
        self.loss_func = SoftmaxCrossEntropyWithLogits(sparse=True)

        self.args = args

        self.num_labels = num_labels
        self.dropout = nn.Dropout(1 - args.hidden_dropout_prob)
        self.classifier = nn.Dense(args.hidden_size * 2, num_labels, has_bias=False)
        self.op = ops.Concat(axis=1)

    def construct(
            self,
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
            label=None,
    ):
        encoder_outputs = super().construct(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )
        feature_vector = self.op((encoder_outputs[1][:, 0, :], encoder_outputs[1][:, 1, :]))
        feature_vector = self.dropout(feature_vector)

        logits = self.classifier(feature_vector)
        return logits
