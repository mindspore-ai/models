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

"""
Define the network of PFNN
A penalty-free neural network method for solving a class of
second-order boundary-value problems on complex geometries
"""
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.common.initializer import Normal


class LenFac(nn.Cell):
    """
    Caclulate the length

    Args:
        bounds: Boundary of area
    """
    def __init__(self, bounds, mu):
        super(LenFac, self).__init__()
        self.bounds = bounds
        self.hx = self.bounds[0, 1] - self.bounds[0, 0]
        self.mu = mu

    def cal_l(self, x):
        """caclulate function"""
        return 1.0 - (1.0 - (x - self.bounds[0, 0]) / self.hx) ** self.mu

    def construct(self, x):
        """forward"""
        return self.cal_l(x)


class NetG(nn.Cell):
    """NetG"""
    def __init__(self):
        super(NetG, self).__init__()
        self.sin = ops.Sin()
        self.fc0 = nn.Dense(2, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc1 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc2 = nn.Dense(10, 1, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.w_tensor = Tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], mstype.float32)
        self.w = Parameter(self.w_tensor, name="w", requires_grad=False)
        self.matmul = nn.MatMul()

    def network_without_label(self, x):
        """caclulate without label"""
        z = self.matmul(x, self.w)
        h = self.sin(self.fc0(x))
        x = self.sin(self.fc1(h)) + z
        return self.fc2(x)

    def network_with_label(self, x, label):
        """caclulate with label"""
        x = self.network_without_label(x)
        return ((x - label)**2).mean()

    def construct(self, x, label=None):
        """forward"""
        if label is None:
            return self.network_without_label(x)
        return self.network_with_label(x, label)


class NetF(nn.Cell):
    """NetF"""
    def __init__(self):
        super(NetF, self).__init__()
        self.sin = ops.Sin()
        self.fc0 = nn.Dense(2, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc1 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc2 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc3 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc4 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc5 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc6 = nn.Dense(10, 1, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.w_tensor = Tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], mstype.float32)
        self.w = Parameter(self.w_tensor, name="w", requires_grad=False)
        self.matmul = nn.MatMul()

    def construct(self, x):
        """forward"""
        z = self.matmul(x, self.w)
        h = self.sin(self.fc0(x))
        x = self.sin(self.fc1(h)) + z
        h = self.sin(self.fc2(x))
        x = self.sin(self.fc3(h)) + x
        h = self.sin(self.fc4(x))
        x = self.sin(self.fc5(h)) + x
        return self.fc6(x)


class Loss(nn.Cell):
    """NetLoss"""
    def __init__(self, net):
        super(Loss, self).__init__()
        self.matmul = nn.MatMul()
        self.grad = ops.composite.GradOperation()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.net = net

    def get_variable(self, InSet_g, InSet_l, InSet_gx, InSet_lx, InSet_a, InSet_size,
                     InSet_dim, InSet_area, InSet_c, BdSet_nlength, BdSet_nr, BdSet_nl, BdSet_ng):
        """Get Parameters for NetLoss"""
        self.InSet_size = InSet_size
        self.InSet_dim = InSet_dim
        self.InSet_area = InSet_area
        self.BdSet_nlength = BdSet_nlength

        self.InSet_g = Parameter(
            Tensor(InSet_g, mstype.float32), name="InSet_g", requires_grad=False)
        self.InSet_l = Parameter(
            Tensor(InSet_l, mstype.float32), name="InSet_l", requires_grad=False)
        self.InSet_gx = Parameter(
            Tensor(InSet_gx, mstype.float32), name="InSet_gx", requires_grad=False)
        self.InSet_lx = Parameter(
            Tensor(InSet_lx, mstype.float32), name="InSet_lx", requires_grad=False)
        self.InSet_a = Parameter(
            Tensor(InSet_a, mstype.float32), name="InSet_a", requires_grad=False)
        self.InSet_c = Parameter(
            Tensor(InSet_c, mstype.float32), name="InSet_c", requires_grad=False)
        self.BdSet_nr = Parameter(
            Tensor(BdSet_nr, mstype.float32), name="BdSet_nr", requires_grad=False)
        self.BdSet_nl = Parameter(
            Tensor(BdSet_nl, mstype.float32), name="BdSet_nl", requires_grad=False)
        self.BdSet_ng = Parameter(
            Tensor(BdSet_ng, mstype.float32), name="BdSet_ng", requires_grad=False)

    def construct(self, InSet_x, BdSet_x):
        """forward"""
        InSet_f = self.net(InSet_x)
        InSet_fx = self.grad(self.net)(InSet_x)
        InSet_u = self.InSet_g + self.InSet_l * InSet_f
        InSet_ux = self.InSet_gx + self.InSet_lx * InSet_f + self.InSet_l * InSet_fx
        InSet_aux = self.matmul(self.InSet_a, InSet_ux.reshape(
            (self.InSet_size, self.InSet_dim, 1)))
        InSet_aux = InSet_aux.reshape(self.InSet_size, self.InSet_dim)
        BdSet_nu = self.BdSet_ng + self.BdSet_nl * self.net(BdSet_x)
        return 0.5 * self.InSet_area * self.sum(self.mean((InSet_aux * InSet_ux), 0)) + \
            self.InSet_area * self.mean(self.InSet_c * InSet_u) - \
            self.BdSet_nlength * self.mean(self.BdSet_nr * BdSet_nu)
