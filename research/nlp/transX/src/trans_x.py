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
TransE/TransD/TransH/TransR models.
"""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Constant
from mindspore.common.initializer import XavierUniform


@ops.constexpr
def tensor_constant(value, shape):
    """Create a tensor constant of the specified shape"""
    constant = np.ones(shape, np.float32) * value
    return Tensor(constant, mstype.float32)


class TransXModel(nn.Cell):
    """Base class for TransX models

    Args:
        ent_tot (int): Size of the entities dictionary.
        rel_tot (int): Size of the relations dictionary.
        dim_e (int): Embeddings size for head/tail entities.
        dim_r (int): Embeddings size for relations.
    """

    def __init__(self, ent_tot, rel_tot, dim_e, dim_r):
        super().__init__()

        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim_e = dim_e
        self.dim_r = dim_r

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e, embedding_table=XavierUniform())
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r, embedding_table=XavierUniform())

    @staticmethod
    def l2norm_op(x):
        """Custom L2 normalization operator"""
        # Default L2 normalization works incredibly slow (2 times than the following implementation)
        # The secret is in the broadcasting. So we are avoiding it in all cost.
        ones_for_sum = tensor_constant(1, (x.shape[1], x.shape[1]))
        eps = tensor_constant(1e-12, x.shape)
        return x * ops.Rsqrt()(ops.matmul(ops.square(x), ones_for_sum) + eps)

    def get_embeddings(self, batched_triplets):
        """Get embeddings for the provided triplet"""
        h = self.ent_embeddings(batched_triplets[:, 0])
        r = self.rel_embeddings(batched_triplets[:, 1])
        t = self.ent_embeddings(batched_triplets[:, 2])
        return h, r, t

    def normalize_embeddings(self, h, r, t):
        """Normalize embeddings"""
        h = self.l2norm_op(h)
        r = self.l2norm_op(r)
        t = self.l2norm_op(t)
        return h, r, t

    @staticmethod
    def calc_score(h, r, t):
        """Calculates the score for the embeddings"""
        score = h + (r - t)
        # Default Reduce sum works slower than the following implementation.
        # The secret is in the broadcasting. So we are avoiding it in all cost.
        ones_for_sum = tensor_constant(1, (score.shape[1], 1))
        # ones_for_sum = ops.Ones()((score.shape[1], 1), mstype.float32)
        score = ops.matmul(ops.absolute(score), ones_for_sum).flatten()
        return score

    def construct(self, batched_triplets):
        """Feed forward"""
        raise NotImplementedError(
            'The base TransX class cannot be used '
            'for building the computational Graph.'
        )


class TransE(TransXModel):
    """TransE model definition

    Args:
        ent_tot (int): Size of the entities dictionary.
        rel_tot (int): Size of the relations dictionary.
        dim (int): Embeddings size.

    Returns:
        Tensor, Model scores for the provided triplets.

    Examples:
        >>> TransE(1000, 1000, dim=100)
    """

    def __init__(
            self,
            ent_tot,
            rel_tot,
            dim=100,
    ):
        super().__init__(ent_tot, rel_tot, dim_e=dim, dim_r=dim)

    def construct(self, batched_triplets):
        """Feed forward"""
        h, r, t = self.get_embeddings(batched_triplets)
        h, r, t = self.normalize_embeddings(h, r, t)
        score = self.calc_score(h, r, t)
        return score


class TransH(TransE):
    """TransH model definition

    Args:
        ent_tot (int): Size of the entities dictionary.
        rel_tot (int): Size of the relations dictionary.
        dim (int): Embeddings size.

    Returns:
        Tensor, Model scores for the provided triplets.

    Examples:
        >>> TransH(1000, 1000, dim=100)
    """

    def __init__(
            self,
            ent_tot,
            rel_tot,
            dim=100,
    ):
        super().__init__(ent_tot, rel_tot, dim=dim)

        self.norm_vector = nn.Embedding(self.rel_tot, self.dim_e, embedding_table=XavierUniform())

    @staticmethod
    def transfer_to_hyperplane(e, normal):
        """Transfer the embedding to the hyperplane defined by the normal"""
        # Default ReduceSum with the following broadcasting works incredibly slow
        # (2 times than the following implementation)
        # The secret is in the broadcasting. So we are avoiding it in all cost.
        ones_for_sum = tensor_constant(1, (e.shape[1], e.shape[1]))
        return e - ops.matmul(e * normal, ones_for_sum) * normal

    def construct(self, batched_triplets):
        """Feed forward"""
        h, r, t = self.get_embeddings(batched_triplets)
        r_norm = self.norm_vector(batched_triplets[:, 1])
        r_norm = self.l2norm_op(r_norm)

        h = self.transfer_to_hyperplane(h, r_norm)
        t = self.transfer_to_hyperplane(t, r_norm)

        h, r, t = self.normalize_embeddings(h, r, t)

        score = self.calc_score(h, r, t)
        return score


class TransR(TransXModel):
    """TransR model definition

    Args:
        ent_tot (int): Size of the entities dictionary.
        rel_tot (int): Size of the relations dictionary.
        dim_e (int): Embeddings size for head/tail entities.
        dim_r (int): Embeddings size for relations.

    Returns:
        Tensor, Model scores for the provided triplets.

    Examples:
        >>> TransR(1000, 1000, dim_e=100, dim_r=100)
    """

    def __init__(
            self,
            ent_tot,
            rel_tot,
            dim_e=100,
            dim_r=100,
    ):
        super().__init__(ent_tot, rel_tot, dim_e=dim_e, dim_r=dim_r)
        identity_weights = np.zeros(self.dim_e * self.dim_r, np.float32)
        np.fill_diagonal(identity_weights.reshape(self.dim_e, self.dim_r), 1)
        init_weights = identity_weights[None].repeat(self.rel_tot, axis=0)
        self.transfer_matrix = nn.Embedding(
            self.rel_tot,
            self.dim_e * self.dim_r,
            embedding_table=Constant(init_weights),
        )

    def apply_transfer_matrix(self, e, transfer_matrix):
        """Transfer the embedding to the hyperplane defined by the normal"""
        transfer_matrix = transfer_matrix.view(-1, self.dim_r, self.dim_e)
        e = e.view(-1, self.dim_e, 1)
        return ops.BatchMatMul()(transfer_matrix, e)[..., 0]

    def construct(self, batched_triplets):
        """Feed forward"""
        h, r, t = self.get_embeddings(batched_triplets)
        r_transfer = self.transfer_matrix(batched_triplets[:, 1])

        h = self.apply_transfer_matrix(h, r_transfer)
        t = self.apply_transfer_matrix(t, r_transfer)

        h, r, t = self.normalize_embeddings(h, r, t)

        score = self.calc_score(h, r, t)
        return score


class TransD(TransXModel):
    """TransD model definition

    Args:
        ent_tot (int): Size of the entities dictionary.
        rel_tot (int): Size of the relations dictionary.
        dim_e (int): Embeddings size for head/tail entities.
        dim_r (int): Embeddings size for relations.

    Returns:
        Tensor, Model scores for the provided triplets.

    Examples:
        >>> TransD(1000, 1000, dim_e=100, dim_r=100)
    """

    def __init__(
            self,
            ent_tot,
            rel_tot,
            dim_e=100,
            dim_r=100,
    ):
        super().__init__(ent_tot, rel_tot, dim_e=dim_e, dim_r=dim_r)
        self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e, embedding_table=XavierUniform())
        self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r, embedding_table=XavierUniform())
        self.reduce = ops.ReduceSum(keep_dims=True)
        if self.dim_r > self.dim_e:
            self.zero_pad_op = ops.Pad(((0, 0), (0, self.dim_r - self.dim_e)))
        else:
            self.zero_pad_op = None

    def transfer(self, e, e_transfer, r_transfer):
        """Map the embeddings onto the relations space

        Mapping matrix M[i,j] = r_transfer[i] * e_transfer[j] + delta[i, j]
        res[i] = sum(M[i,j] * e[j]; j)
        """
        dynamic_projection = self.reduce(e * e_transfer, -1) * r_transfer

        if self.dim_r < self.dim_e:
            # If the relations embeddings size is smaller than the size
            # of head/tail embeddings, the excess is removed.
            identity_projection = e[..., :self.dim_r]
        elif self.dim_r > self.dim_e:
            # If the relations embeddings size is larger than the size
            # of head/tail embeddings, the later are padded.
            identity_projection = self.zero_pad_op(e)
        else:
            # If dimensions are the same, then no transformation applies.
            identity_projection = e

        return identity_projection + dynamic_projection

    def _get_transfer_matrices(self, batched_triplets):
        """Get transfer matrices"""
        b_h = batched_triplets[:, 0]
        b_r = batched_triplets[:, 1]
        b_t = batched_triplets[:, 2]
        h_transfer = self.ent_transfer(b_h)
        r_transfer = self.rel_transfer(b_r)
        t_transfer = self.ent_transfer(b_t)
        return h_transfer, r_transfer, t_transfer

    def construct(self, batched_triplets):
        """Feed forward"""
        h, r, t = self.get_embeddings(batched_triplets)
        h_transfer, r_transfer, t_transfer = self._get_transfer_matrices(batched_triplets)

        h = self.transfer(h, h_transfer, r_transfer)
        t = self.transfer(t, t_transfer, r_transfer)

        h, r, t = self.normalize_embeddings(h, r, t)

        score = self.calc_score(h, r, t)
        return score
