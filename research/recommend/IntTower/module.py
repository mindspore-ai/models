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


from mindspore import nn, ops
from mindspore.ops import L2Normalize, broadcast_to


class LightSE(nn.Cell):
    """LightSELayer used in IntTower.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **seed** : A Python integer to use as random seed.
      References
      """

    def __init__(self, field_size, seed=1024):
        super(LightSE, self).__init__()
        self.seed = seed
        self.softmax = nn.Softmax(axis=1)
        self.field_size = field_size
        self.excitation = nn.Dense(self.field_size, self.field_size)

    def construct(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        Z = ops.mean(inputs, axis=-1)
        A = self.excitation(Z)  # (batch,reduction_size)
        A = self.softmax(A)  # (batch,reduction_size)
        out = inputs * ops.expand_dims(A, axis=2)

        return inputs + out


class ContrastLoss(nn.LossBase):
    def __init__(self, reduction="mean"):
        """compute contrast loss
              Input shape
        - Two 2D tensors with shape: ``(batch_size,embedding_size)``.
      Output shape
        - A loss scalar.

        """

        super(ContrastLoss, self).__init__(reduction)
        self.norm = L2Normalize(axis=-1)
        self.cos_sim = nn.CosineEmbeddingLoss()
        self.abs = ops.Abs()
        self.lam = 1
        self.pos = 0
        self.all = 0
        self.tau = 1

    def construct(self, user_embedding, item_embedding, target):
        user_embedding = self.norm(user_embedding)
        item_embedding = self.norm(item_embedding)
        pos_index = broadcast_to(target, (target.shape[0], item_embedding.shape[1]))
        self.pos += self.abs(ops.mean(user_embedding * item_embedding * pos_index)) / self.tau
        self.all += self.abs(ops.mean(user_embedding * item_embedding)) / self.tau
        contrast_loss = -ops.log(ops.exp(self.pos) / ops.exp(self.all)) * self.lam
        return contrast_loss
