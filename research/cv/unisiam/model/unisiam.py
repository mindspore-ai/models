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
import math
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as np


class UniSiam(ms.nn.Cell):
    def __init__(self, encoder, lamb=0.1, temp=2.0, dim_hidden=None, dist=False, dim_out=2048):
        super().__init__()
        self.encoder = encoder
        self.encoder.fc = None

        dim_in = encoder.num_features
        dim_hidden = dim_in if dim_hidden is None else dim_hidden

        self.proj = ms.nn.SequentialCell([
            ms.nn.Dense(dim_in, dim_hidden),
            ms.nn.BatchNorm1d(dim_hidden),
            ms.nn.ReLU(),
            ms.nn.Dense(dim_hidden, dim_hidden),
            ms.nn.BatchNorm1d(dim_hidden),
            ms.nn.ReLU(),
            ms.nn.Dense(dim_hidden, dim_hidden),
            ms.nn.BatchNorm1d(dim_hidden),])
        self.pred = ms.nn.SequentialCell([
            ms.nn.Dense(dim_hidden, dim_hidden//4),
            ms.nn.BatchNorm1d(dim_hidden//4),
            ms.nn.ReLU(),
            ms.nn.Dense(dim_hidden//4, dim_hidden)])

        if dist:
            self.pred_dist = ms.nn.SequentialCell([
                ms.nn.Dense(dim_in, dim_out),
                ms.nn.BatchNorm1d(dim_out),
                ms.nn.ReLU(),
                ms.nn.Dense(dim_out, dim_out),
                ms.nn.BatchNorm1d(dim_out),
                ms.nn.ReLU(),
                ms.nn.Dense(dim_out, dim_out),
                ms.nn.BatchNorm1d(dim_out),
                ms.nn.ReLU(),
                ms.nn.Dense(dim_out, dim_out//4),
                ms.nn.BatchNorm1d(dim_out//4),
                ms.nn.ReLU(),
                ms.nn.Dense(dim_out//4, dim_out)])

        self.lamb = lamb
        self.temp = temp

        for _, cell in self.cells_and_names():
            if isinstance(cell, (ms.nn.BatchNorm2d)):
                cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, (ms.nn.Dense)):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
                    cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def construct(self, x, z_dist=None):

        f = self.encoder(x)
        z = self.proj(f)
        p = self.pred(z)
        z1, z2 = np.split(z, axis=0, indices_or_sections=2)
        p1, p2 = np.split(p, axis=0, indices_or_sections=2)

        loss_pos = (self.pos(p1, z2)+self.pos(p2, z1))/2
        loss_neg = self.neg(z)
        loss = loss_pos + self.lamb * loss_neg

        if z_dist is not None:
            p_dist = self.pred_dist(f)
            loss_dist = self.pos(p_dist, z_dist)
            loss = 0.5 * loss + 0.5 * loss_dist

        std = self.std(z)

        return loss, loss_pos, loss_neg, std

    def std(self, z):
        return (z/np.norm(z, axis=1, keepdims=True)).std(axis=0).mean()

    def pos(self, p, z):
        z = ops.stop_gradient(z)
        z /= np.norm(z, axis=1, keepdims=True)
        p /= np.norm(p, axis=1, keepdims=True)
        return  -(p*z).sum(axis=1).mean()


    def neg(self, z):
        batch_size = z.shape[0] //2
        n_neg = z.shape[0] - 2
        z /= np.norm(z, axis=-1, keepdims=True)
        mask = 1-ops.eye(batch_size, batch_size, ms.float32)
        mask = np.tile(mask, (2, 2))
        out = np.matmul(z, z.T) * mask
        return np.log(np.mean((np.exp(out/self.temp).sum(axis=1)-2)/n_neg))
