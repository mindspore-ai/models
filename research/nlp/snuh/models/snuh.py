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

"""SNUH model."""
import mindspore
from mindspore import nn, ops, Tensor
from mindspore import numpy as mindnp

class SNUH(nn.Cell):
    """SNUH class"""
    def __init__(self, hparams, num_nodes, num_edges, vocab_size):
        """init method"""
        super(SNUH, self).__init__()
        self.hparams = hparams
        self.num_nodes, self.num_edges = num_nodes, num_edges

        self.venc = VarEncoder(vocab_size, self.hparams.num_features)
        self.cenc = CorrEncoder(vocab_size, self.hparams.num_features)
        self.dec = Decoder(self.hparams.num_features, vocab_size)

        self.stdnormal_ = ops.StandardNormal()
        self.sign_ = ops.Sign()
        self.log_ = ops.Log()
        self.round_ = ops.Round()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.tau = Tensor(0.99, mindspore.float32)
        self.temperature = Tensor(self.hparams.temperature, mindspore.float32)
        self.beta = Tensor(self.hparams.beta, mindspore.float32)

    def construct(self, x, label, edge1, edge2, weight):
        """model construct method"""
        q_mu, q_sigma = self.venc(x, self.temperature)

        eps = self.stdnormal_(q_mu.shape)
        z_st = q_mu + q_sigma * eps

        log_likelihood = self.dec(z_st, self.sign_(x))
        kl = self.compute_kl(q_mu, q_sigma, edge1, edge2, weight, self.num_nodes, self.num_edges)

        loss = -log_likelihood + self.beta * kl
        return loss

    def compute_kl(self, q_mu, q_sigma, edge1, edge2, weight, num_node, num_edges):
        """compute kl divergence"""
        q_mu1, q_sigma1 = self.venc(edge1, self.temperature)
        q_mu2, q_sigma2 = self.venc(edge2, self.temperature)

        kl_node = self.reduce_mean(self.reduce_sum(q_mu**2 + q_sigma**2 - 1 - 2*self.log_(q_sigma + 1e-8), 1))

        gamma = self.cenc(edge1, edge2)
        kl_edge = self.reduce_mean(self.reduce_sum(0.5 *\
                    (q_mu1**2 + q_mu2**2 + q_sigma1**2 + q_sigma2**2 - 2 * self.tau * gamma * q_sigma1 *\
                    q_sigma2 - 2 * self.tau * q_mu1 * q_mu2)\
                    / (1 - self.tau**2) - 0.5 * (q_mu1**2 + q_mu2**2 + q_sigma1**2 + q_sigma2**2)\
                    - 0.5 * self.log_(1 - gamma + 1e-8) + 0.5 * self.log_(1 - self.tau**2), 1) * weight)
        return kl_node + kl_edge * num_edges / num_node

    def encode_discrete(self, x):
        mu, _ = self.venc(x, self.temperature)
        return (mu > 0.5).astype(mindspore.float32)

class VarEncoder(nn.Cell):
    """variational encoder"""
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.dim_output = dim_output
        self.ff = nn.Dense(dim_input, 2 * dim_output)
        self.sigmoid_ = ops.Sigmoid()
        self.log_ = ops.Log()
        self.exp_ = ops.Exp()

    def construct(self, x, temperature):
        gaussian_params = self.ff(x)
        mu = self.sigmoid_(gaussian_params[:, :self.dim_output] / temperature)
        sigma = self.log_(1 + self.exp_(gaussian_params[:, self.dim_output:]))  # softplus
        return mu, sigma

class CorrEncoder(nn.Cell):
    """correlated encoder"""
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.dim_output = dim_output
        self.ff = nn.Dense(2 * dim_input, dim_output)
        self.sigmoid_ = ops.Sigmoid()

    def construct(self, x1, x2):
        net = mindnp.concatenate([mindnp.concatenate([x1, x2], axis=1), mindnp.concatenate([x2, x1], axis=1)], axis=0)
        corr_params = mindnp.reshape(self.ff(net), [2, -1, self.dim_output])
        corr_params = (corr_params[0] + corr_params[1]) / 2.0
        correlation_coefficient = (1. - 1e-8) * (2. * self.sigmoid_(corr_params) - 1.)
        return correlation_coefficient

class Decoder(nn.Cell):  # As in VDSH, NASH, BMSH
    """generative decoder"""
    def __init__(self, dim_encoding, vocab_size):
        super().__init__()
        self.ff = nn.Dense(dim_encoding, vocab_size)
        self.log_softmax = nn.LogSoftmax(axis=1)
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, z, targets):
        scores = self.ff(z)
        log_probs = self.log_softmax(scores)
        log_likelihood = self.reduce_mean(self.reduce_sum(log_probs * targets, 1))
        return log_likelihood
