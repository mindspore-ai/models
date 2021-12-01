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
""" layer ms """


import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from model_mindspore.parallel_transformer import LayerNorm2


class Linear(nn.Cell):
    """ Linear """

    def __init__(self):
        super().__init__()
        self.matmul = nn.MatMul()
        self.t = P.Transpose()

    def construct(self, input1, weight, bias=None):
        output = self.matmul(input1, self.t(weight, (1, 0)))
        if bias is not None:
            output += bias
        return output


# for MRM
class RegionFeatureRegression(nn.Cell):
    """ RegionFeatureRegression """

    def __init__(self, config, hidden_size, feat_dim, img_linear_weight, parallel_config):
        super().__init__()
        self.dense = nn.Dense(hidden_size, hidden_size).to_float(mindspore.float16)
        self.dense.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.dense.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.dense.weight.parallel_optimizer = False
        self.dense.bias.parallel_optimizer = False
        self.gelu = nn.GELU()
        self.gelu.gelu.shard(((parallel_config.dp, 1),))
        self.layer_norm = LayerNorm2((hidden_size,), parallel_config.dp).to_float(mindspore.float32)

        self.weight = img_linear_weight
        self.bias = Parameter(Tensor(np.zeros(feat_dim), mstype.float32), parallel_optimizer=False)
        self.add = P.TensorAdd().shard(((parallel_config.dp, 1), (1,)))
        self.matmul = P.MatMul().shard(((parallel_config.dp, 1), (1, 1)))

    def construct(self, input_):
        dense_out = self.dense(input_)
        gelu_out = self.gelu(dense_out)
        hidden = self.layer_norm(gelu_out)
        output = self.add(self.matmul(hidden.astype(mstype.float16), self.weight.astype(mstype.float16)),
                          self.bias.astype(mstype.float16))

        return output.astype(mindspore.float32)


# for MAR
class AudioFeatureRegression(nn.Cell):
    """ AudioFeatureRegression """

    def __init__(self, config, hidden_size, feat_dim, img_linear_weight, parallel_config):
        super().__init__()
        self.dense = nn.Dense(hidden_size, hidden_size).to_float(mindspore.float16)
        self.dense.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.dense.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.dense.weight.parallel_optimizer = False
        self.dense.bias.parallel_optimizer = False
        self.gelu = nn.GELU()
        self.gelu.gelu.shard(((parallel_config.dp, 1),))
        self.layer_norm = LayerNorm2((hidden_size,), parallel_config.dp).to_float(mindspore.float32)

        self.weight = img_linear_weight
        self.bias = Parameter(Tensor(np.zeros(feat_dim), mstype.float32), parallel_optimizer=False)

        self.add = P.TensorAdd().shard(((parallel_config.dp, 1), (1,)))
        self.matmul = P.MatMul().shard(((parallel_config.dp, 1), (1, 1)))

    def construct(self, input_):
        dense_out = self.dense(input_)
        gelu_out = self.gelu(dense_out)
        hidden = self.layer_norm(gelu_out)
        output = self.add(self.matmul(hidden.astype(mstype.float16), self.weight.astype(mstype.float16)),
                          self.bias.astype(mstype.float16))
        return output.astype(mindspore.float32)


# for MRC(-kl)
class RegionClassification(nn.Cell):
    """ RegionClassification """

    def __init__(self, config, hidden_size, label_dim, parallel_config):
        super().__init__()
        self.dense_1 = nn.Dense(hidden_size, hidden_size).to_float(mindspore.float16)
        self.dense_1.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.dense_1.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.dense_1.weight.parallel_optimizer = False
        self.dense_1.bias.parallel_optimizer = False
        self.gelu = nn.GELU()
        self.gelu.gelu.shard(((parallel_config.dp, 1),))
        self.layer_norm = LayerNorm2((hidden_size,), parallel_config.dp).to_float(mindspore.float32)
        self.dense_2 = nn.Dense(hidden_size, label_dim).to_float(mindspore.float16)
        self.dense_2.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.dense_2.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.dense_2.weight.parallel_optimizer = False
        self.dense_2.bias.parallel_optimizer = False

    def construct(self, input_):
        dense_1_out = self.dense_1(input_)
        gelu_out = self.gelu(dense_1_out)
        layer_norm_out = self.layer_norm(gelu_out)
        output = self.dense_2(layer_norm_out)
        return output.astype(mindspore.float32)


class AudioClassification(nn.Cell):
    """ AudioClassification """

    def __init__(self, hidden_size, label_dim, parallel_config):
        super().__init__()
        self.dense_1 = nn.Dense(hidden_size, hidden_size).to_float(mindspore.float16)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm([hidden_size], epsilon=1e-12).to_float(mindspore.float32)
        self.dense_2 = nn.Dense(hidden_size, label_dim).to_float(mindspore.float16)

    def construct(self, input_):
        dense_1_out = self.dense_1(input_)
        gelu_out = self.gelu(dense_1_out)
        layer_norm_out = self.layer_norm(gelu_out)
        output = self.dense_2(layer_norm_out)
        return output.astype(mindspore.float32)


class BertOnlyMLMHead(nn.Cell):
    """ BertOnlyMLMHead """

    def __init__(self, config, bert_model_embedding_weights, parallel_config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights, parallel_config)

    def construct(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertLMPredictionHead(nn.Cell):
    """ BertLMPredictionHead """

    def __init__(self, config, bert_model_embedding_weights, parallel_config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config, parallel_config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(bert_model_embedding_weights.shape[1],
                                bert_model_embedding_weights.shape[0],
                                has_bias=True).to_float(mindspore.float16)
        self.decoder.weight = bert_model_embedding_weights
        self.decoder.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.decoder.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.decoder.weight.parallel_optimizer = False
        self.decoder.bias.parallel_optimizer = False

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states.astype(mindspore.float32)


class BertPredictionHeadTransform(nn.Cell):
    """ BertPredictionHeadTransform """

    def __init__(self, config, parallel_config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size).to_float(mindspore.float16)
        self.dense.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.dense.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.dense.weight.parallel_optimizer = False
        self.dense.bias.parallel_optimizer = False
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = nn.ReLU()
            self.transform_act_fn.relu.shard(((parallel_config.dp, 1),))
        else:
            self.transform_act_fn = nn.GELU()
            self.transform_act_fn.gelu.shard(((parallel_config.dp, 1),))
        self.LayerNorm = LayerNorm2((config.hidden_size,), parallel_config.dp).to_float(mindspore.float32)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
