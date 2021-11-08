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
"""Data for PFNN"""
import numpy as np

def uu_prob1(x, order):
    """provide label about prob1"""
    temp0 = x[:, 0:1] + x[:, 1:2]
    temp1 = x[:, 0:1] - x[:, 1:2]
    temp2 = 10 * temp0 ** 2 + temp1 ** 2 + 0.5
    if order[0] == 0 and order[1] == 0:
        return np.log(temp2)
    if order[0] == 1 and order[1] == 0:
        return temp2 ** (-1) * (20 * temp0 + 2 * temp1)
    if order[0] == 0 and order[1] == 1:
        return temp2 ** (-1) * (20 * temp0 - 2 * temp1)
    if order[0] == 2 and order[1] == 0:
        return -temp2 ** (-2) * (20 * temp0 + 2 * temp1) ** 2 \
            + temp2 ** (-1) * (22)
    if order[0] == 1 and order[1] == 1:
        return - temp2 ** (-2) * (20 * temp0 + 2 * temp1) * (20 * temp0 - 2 * temp1) \
            + temp2 ** (-1) * (18)
    if order[0] == 0 and order[1] == 2:
        return - temp2 ** (-2) * (20 * temp0 - 2 * temp1) ** 2 \
            + temp2 ** (-1) * (22)
    return None

def uu_prob2(x, order):
    """provide label about prob2"""
    temp0 = np.exp(2 * x[:, 1:2])
    temp1 = np.exp(-2 * x[:, 1:2])
    if order[0] == 0 and order[1] == 0:
        return (x[:, 0:1] ** 3 - x[:, 0:1]) * 0.5 * (temp0 + temp1)
    if order[0] == 1 and order[1] == 0:
        return (3 * x[:, 0:1] ** 2 - 1) * 0.5 * (temp0 + temp1)
    if order[0] == 0 and order[1] == 1:
        return (x[:, 0:1] ** 3 - x[:, 0:1]) * (temp0 - temp1)
    if order[0] == 2 and order[1] == 0:
        return (6 * x[:, 0:1]) * 0.5 * (temp0 + temp1)
    if order[0] == 1 and order[1] == 1:
        return (3 * x[:, 0:1] ** 2 - 1) * (temp0 - temp1)
    if order[0] == 0 and order[1] == 2:
        return (x[:, 0:1] ** 3 - x[:, 0:1]) * 2 * (temp0 + temp1)
    return None

def uu_prob3(x, order):
    """provide label about prob3"""
    temp0 = x[:, 0:1]*x[:, 0:1] - x[:, 1:2] * x[:, 1:2]
    temp1 = x[:, 0:1]*x[:, 0:1] + x[:, 1:2] * x[:, 1:2] + 0.1
    if order[0] == 0 and order[1] == 0:
        return temp0 * temp1 ** (-1)
    if order[0] == 1 and order[1] == 0:
        return (2 * x[:, 0:1]) * temp1 ** (-1) + \
            temp0 * (-1) * temp1 ** (-2) * (2 * x[:, 0:1])
    if order[0] == 0 and order[1] == 1:
        return (-2 * x[:, 1:2]) * temp1 ** (-1) + \
            temp0 * (-1) * temp1 ** (-2) * (2 * x[:, 1:2])
    if order[0] == 2 and order[1] == 0:
        return (2) * temp1 ** (-1) + \
            2 * (2 * x[:, 0:1]) * (-1) * temp1 ** (-2) * (2 * x[:, 0:1]) + \
            temp0 * (2) * temp1 ** (-3) * (2 * x[:, 0:1]) ** 2 + \
            temp0 * (-1) * temp1 ** (-2) * (2)
    if order[0] == 1 and order[1] == 1:
        return (2 * x[:, 0:1]) * (-1) * temp1 ** (-2) * (2 * x[:, 1:2]) + \
                (-2 * x[:, 1:2]) * (-1) * temp1 ** (-2) * (2 * x[:, 0:1]) + \
            temp0 * (2)*temp1 ** (-3) * (2 * x[:, 0:1]) * (2 * x[:, 1:2])
    if order[0] == 0 and order[1] == 2:
        return (-2) * temp1 ** (-1) + \
            2 * (-2 * x[:, 1:2]) * (-1) * temp1 ** (-2) * (2 * x[:, 1:2]) + \
            temp0 * (2) * temp1 ** (-3) * (2 * x[:, 1:2]) ** 2 + \
            temp0 * (-1) * temp1 ** (-2) * (2)
    return None

def uu_prob4(x, order):
    """provide label about prob4"""
    temp = np.exp(-4 * x[:, 1:2] * x[:, 1:2])
    if order[0] == 0 and order[1] == 0:
        ind = (x[:, 0:1] <= 0).float()
        return ind * ((x[:, 0:1] + 1) ** 4 - 1) * temp + \
            (1 - ind) * (-(-x[:, 0:1] + 1) ** 4 + 1) * temp
    if order[0] == 1 and order[1] == 0:
        ind = (x[:, 0:1] <= 0).float()
        return ind * (4 * (x[:, 0:1] + 1) ** 3) * temp + \
            (1 - ind) * (4 * (-x[:, 0:1] + 1) ** 3) * temp
    if order[0] == 0 and order[1] == 1:
        ind = (x[:, 0:1] <= 0).float()
        return ind * ((x[:, 0:1] + 1) ** 4 - 1) * (temp * (-8 * x[:, 1:2])) + \
            (1 - ind) * (-(- x[:, 0:1] + 1) ** 4 + 1) * (temp * (- 8 * x[:, 1:2]))
    if order[0] == 2 and order[1] == 0:
        ind = (x[:, 0:1] <= 0).float()
        return ind * (12 * (x[:, 0:1] + 1) ** 2) * temp + \
            (1 - ind) * (-12 * (-x[:, 0:1] + 1) ** 2) * temp
    if order[0] == 1 and order[1] == 1:
        ind = (x[:, 0:1] <= 0).float()
        return ind * (4 * (x[:, 0:1] + 1) ** 3) * (temp * (-8 * x[:, 1:2])) + \
            (1 - ind) * (4 * (-x[:, 0:1] + 1) ** 3) * (temp * (-8 * x[:, 1:2]))
    if order[0] == 0 and order[1] == 2:
        ind = (x[:, 0:1] <= 0).float()
        return ind * ((x[:, 0:1] + 1) ** 4 - 1) * (temp * (64 * x[:, 1:2] * x[:, 1:2] - 8)) + \
            (1 - ind) * (-(-x[:, 0:1] + 1) ** 4 + 1) * \
            (temp * (64 * x[:, 1:2] * x[:, 1:2] - 8))
    return None

def uu(x, order, prob):
    """
    Provide label/true answer

    Args:
        prob: the type of problem
    """
    if prob == 1:
        return uu_prob1(x, order)
    if prob == 2:
        return uu_prob2(x, order)
    if prob == 3:
        return uu_prob3(x, order)
    if prob == 4:
        return uu_prob4(x, order)
    return None

def aa_00(x, ind):
    """Function A with order 0,0"""
    if ind[0] == 0 and ind[1] == 0:
        return (x[:, 0:1] + x[:, 1:2]) * (x[:, 0:1] + x[:, 1:2]) + 1
    if ind[0] == 0 and ind[1] == 1:
        return - (x[:, 0:1] + x[:, 1:2]) * (x[:, 0:1] - x[:, 1:2])
    if ind[0] == 1 and ind[1] == 0:
        return - (x[:, 0:1] + x[:, 1:2]) * (x[:, 0:1] - x[:, 1:2])
    if ind[0] == 1 and ind[1] == 1:
        return (x[:, 0:1] - x[:, 1:2]) * (x[:, 0:1] - x[:, 1:2]) + 1
    return None

def aa_10(x, ind):
    """Function A with order 1,0"""
    if ind[0] == 0 and ind[1] == 0:
        return 2 * (x[:, 0:1] + x[:, 1:2])
    if ind[0] == 0 and ind[1] == 1:
        return -2 * x[:, 0:1]
    if ind[0] == 1 and ind[1] == 0:
        return -2 * x[:, 0:1]
    if ind[0] == 1 and ind[1] == 1:
        return 2 * (x[:, 0:1] - x[:, 1:2])
    return None

def aa_01(x, ind):
    """Function A with order 0,1"""
    if ind[0] == 0 and ind[1] == 0:
        return 2 * (x[:, 0:1] + x[:, 1:2])
    if ind[0] == 0 and ind[1] == 1:
        return 2 * x[:, 1:2]
    if ind[0] == 1 and ind[1] == 0:
        return 2 * x[:, 1:2]
    if ind[0] == 1 and ind[1] == 1:
        return -2 * (x[:, 0:1] - x[:, 1:2])
    return None

def aa(x, ind, order):
    """Function A"""
    if order[0] == 0 and order[1] == 0:
        return aa_00(x, ind)
    if order[0] == 1 and order[1] == 0:
        return aa_10(x, ind)
    if order[0] == 0 and order[1] == 1:
        return aa_01(x, ind)
    return None

def cc(x, prob):
    """Function C"""
    return aa(x, [0, 0], [1, 0]) * uu(x, [1, 0], prob) + aa(x, [0, 0], [0, 0]) * uu(x, [2, 0], prob) + \
        aa(x, [1, 0], [1, 0]) * uu(x, [0, 1], prob) + aa(x, [0, 1], [0, 0]) * uu(x, [1, 1], prob) + \
        aa(x, [0, 1], [0, 1]) * uu(x, [1, 0], prob) + aa(x, [1, 0], [0, 0]) * uu(x, [1, 1], prob) + \
        aa(x, [1, 1], [0, 1]) * uu(x, [0, 1], prob) + \
        aa(x, [1, 1], [0, 0]) * uu(x, [0, 2], prob)

def rr_n(x, n, prob):
    """Right hand side of Neumann boundary equation"""
    return (aa(x, [0, 0], [0, 0]) * uu(x, [1, 0], prob) +
            aa(x, [0, 1], [0, 0]) * uu(x, [0, 1], prob)) * n[:, 0:1] + \
           (aa(x, [1, 0], [0, 0]) * uu(x, [1, 0], prob) +
            aa(x, [1, 1], [0, 0]) * uu(x, [0, 1], prob)) * n[:, 1:2]

class InnerSet():
    """Inner Set"""
    def __init__(self, bounds, nx, prob):
        self.dim = 2
        self.bounds = bounds
        self.area = (self.bounds[0, 1] - self.bounds[0, 0]) * \
                    (self.bounds[1, 1] - self.bounds[1, 0])

        self.gp_num = 2
        self.gp_wei = [1.0, 1.0]
        self.gp_pos = [(1 - 0.5773502692) / 2, (1 + 0.5773502692) / 2]

        self.nx = [int(nx[0] / self.gp_num), int(nx[1] / self.gp_num)]
        self.hx = [(self.bounds[0, 1] - self.bounds[0, 0]) / self.nx[0],
                   (self.bounds[1, 1] - self.bounds[1, 0]) / self.nx[1]]

        self.size = self.nx[0] * self.gp_num * self.nx[1] * self.gp_num
        self.x = np.zeros([self.size, self.dim], dtype=np.float32)
        self.a = np.zeros([self.size, self.dim, self.dim], dtype=np.float32)
        self.ax = np.zeros([self.size, self.dim, self.dim], dtype=np.float32)
        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                for k in range(self.gp_num):
                    for l in range(self.gp_num):
                        self.x[m, 0] = self.bounds[0, 0] + \
                            (i + self.gp_pos[k]) * self.hx[0]
                        self.x[m, 1] = self.bounds[1, 0] + \
                            (j + self.gp_pos[l]) * self.hx[1]
                        m = m + 1

        self.a[:, 0, 0:1] = aa(self.x, [0, 0], [0, 0])
        self.a[:, 0, 1:2] = aa(self.x, [0, 1], [0, 0])
        self.a[:, 1, 0:1] = aa(self.x, [1, 0], [0, 0])
        self.a[:, 1, 1:2] = aa(self.x, [1, 1], [0, 0])
        self.ax[:, 0, 0:1] = aa(self.x, [0, 0], [1, 0])
        self.ax[:, 0, 1:2] = aa(self.x, [0, 1], [1, 0])
        self.ax[:, 1, 0:1] = aa(self.x, [1, 0], [0, 1])
        self.ax[:, 1, 1:2] = aa(self.x, [1, 1], [0, 1])
        self.c = cc(self.x, prob)
        self.ua = uu(self.x, [0, 0], prob)


class BoundarySet():
    """Boundary Set"""
    def __init__(self, bounds, nx, prob):
        self.dim = 2
        self.bounds = bounds

        self.d_length = self.bounds[1, 1] - self.bounds[1, 0]
        self.n_length = 2 * (self.bounds[0, 1] - self.bounds[0, 0]) + \
                          (self.bounds[1, 1] - self.bounds[1, 0])

        self.gp_num = 2
        self.gp_wei = [1.0, 1.0]
        self.gp_pos = [(1 - 0.5773502692) / 2, (1 + 0.5773502692) / 2]

        self.nx = [int(nx[0] / self.gp_num), int(nx[1] / self.gp_num)]
        self.hx = [(self.bounds[0, 1] - self.bounds[0, 0]) / self.nx[0],
                   (self.bounds[1, 1] - self.bounds[1, 0]) / self.nx[1]]

        self.d_size = self.nx[1] * self.gp_num
        self.n_size = (2 * self.nx[0] + self.nx[1]) * self.gp_num
        self.d_x = np.zeros([self.d_size, self.dim], dtype=np.float32)
        self.n_x = np.zeros([self.n_size, self.dim], dtype=np.float32)
        self.d_n = np.zeros([self.d_size, self.dim], dtype=np.float32)
        self.n_n = np.zeros([self.n_size, self.dim], dtype=np.float32)
        self.d_a = np.zeros(
            [self.d_size, self.dim, self.dim], dtype=np.float32)
        self.n_a = np.zeros(
            [self.n_size, self.dim, self.dim], dtype=np.float32)
        m = 0
        for j in range(self.nx[1]):
            for l in range(self.gp_num):
                self.d_x[m, 0] = self.bounds[0, 0]
                self.d_x[m, 1] = self.bounds[1, 0] + \
                    (j + self.gp_pos[l]) * self.hx[1]
                self.d_n[m, 0] = -1.0
                self.d_n[m, 1] = 0.0
                m = m + 1
        m = 0
        for i in range(self.nx[0]):
            for k in range(self.gp_num):
                self.n_x[m, 0] = self.bounds[0, 0] + \
                    (i + self.gp_pos[k]) * self.hx[0]
                self.n_x[m, 1] = self.bounds[1, 0]
                self.n_n[m, 0] = 0.0
                self.n_n[m, 1] = -1.0
                m = m + 1
        for j in range(self.nx[1]):
            for l in range(self.gp_num):
                self.n_x[m, 0] = self.bounds[0, 1]
                self.n_x[m, 1] = self.bounds[1, 0] + \
                    (j + self.gp_pos[l]) * self.hx[1]
                self.n_n[m, 0] = 1.0
                self.n_n[m, 1] = 0.0
                m = m + 1
        for i in range(self.nx[0]):
            for k in range(self.gp_num):
                self.n_x[m, 0] = self.bounds[0, 0] + \
                    (i + self.gp_pos[k]) * self.hx[0]
                self.n_x[m, 1] = self.bounds[1, 1]
                self.n_n[m, 0] = 0.0
                self.n_n[m, 1] = 1.0
                m = m + 1

        self.d_a[:, 0, 0:1] = aa(self.d_x, [0, 0], [0, 0])
        self.d_a[:, 0, 1:2] = aa(self.d_x, [0, 1], [0, 0])
        self.d_a[:, 1, 0:1] = aa(self.d_x, [1, 0], [0, 0])
        self.d_a[:, 1, 1:2] = aa(self.d_x, [1, 1], [0, 0])
        self.n_a[:, 0, 0:1] = aa(self.n_x, [0, 0], [0, 0])
        self.n_a[:, 0, 1:2] = aa(self.n_x, [0, 1], [0, 0])
        self.n_a[:, 1, 0:1] = aa(self.n_x, [1, 0], [0, 0])
        self.n_a[:, 1, 1:2] = aa(self.n_x, [1, 1], [0, 0])
        self.d_r = uu(self.d_x, [0, 0], prob)
        self.n_r = rr_n(self.n_x, self.n_n, prob)


class TestSet():
    """Test Set"""
    def __init__(self, bounds, nx, prob):
        self.dim = 2
        self.bounds = bounds
        self.nx = nx
        self.hx = [(self.bounds[0, 1] - self.bounds[0, 0]) / (self.nx[0] - 1),
                   (self.bounds[1, 1] - self.bounds[1, 0]) / (self.nx[1] - 1)]

        self.size = self.nx[0] * self.nx[1]
        self.x = np.zeros([self.size, self.dim], dtype=np.float32)
        m = 0
        for i in range(self.nx[0]):
            for j in range(self.nx[1]):
                self.x[m, 0] = self.bounds[0, 0] + i * self.hx[0]
                self.x[m, 1] = self.bounds[1, 0] + j * self.hx[1]
                m = m + 1

        self.ua = uu(self.x, [0, 0], prob)


def GenerateSet(args):
    """
    Generate Set
    """
    bound = np.array(args.bound).reshape(2, 2)
    problem = args.problem
    return InnerSet(bound, args.inset_nx, problem),\
        BoundarySet(bound, args.bdset_nx, problem), \
        TestSet(bound, args.teset_nx, problem)
