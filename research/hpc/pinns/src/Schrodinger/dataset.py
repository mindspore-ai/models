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
"""Create dataset for training or evaluation"""
import mindspore.dataset as ds
import numpy as np
import scipy.io as scio
from pyDOE import lhs


class PINNs_training_set:
    """
    Training set for PINNs (Schrodinger)

    Args:
        N0 (int): number of sampled training data points for the initial condition
        Nb (int): number of sampled training data points for the boundary condition
        Nf (int): number of sampled training data points for the collocation points
        lb (np.array): lower bound (x, t) of domain
        ub (np.array): upper bound (x, t) of domain
        path (str): path of dataset
    """
    def __init__(self, N0, Nb, Nf, lb, ub, path='./Data/NLS.mat'):
        data = scio.loadmat(path)
        self.N0 = N0
        self.Nb = Nb
        self.Nf = Nf
        self.lb = lb
        self.ub = ub

        # load data
        t = data['tt'].flatten()[:, None]
        x = data['x'].flatten()[:, None]
        Exact = data['uu']
        Exact_u = np.real(Exact)
        Exact_v = np.imag(Exact)

        idx_x = np.random.choice(x.shape[0], self.N0, replace=False)
        self.x0 = x[idx_x, :]
        self.u0 = Exact_u[idx_x, 0:1]
        self.v0 = Exact_v[idx_x, 0:1]

        idx_t = np.random.choice(t.shape[0], self.Nb, replace=False)
        self.tb = t[idx_t, :]

        self.X_f = self.lb + (self.ub-self.lb)*lhs(2, self.Nf)

    def __getitem__(self, index):
        box_0 = np.ones((self.N0, 1), np.float32)
        box_b = np.ones((self.Nb, 1), np.float32)
        box_f = np.ones((self.Nf, 1), np.float32)

        x = np.vstack((self.x0.astype(np.float32),
                       self.lb[0].astype(np.float32) * box_b,
                       self.ub[0].astype(np.float32) * box_b,
                       self.X_f[:, 0:1].astype(np.float32)))
        t = np.vstack((np.array([0], np.float32) * box_0,
                       self.tb.astype(np.float32),
                       self.tb.astype(np.float32),
                       self.X_f[:, 1:2].astype(np.float32)))
        u_target = np.vstack((self.u0.astype(np.float32),
                              self.ub[0].astype(np.float32) * box_b,
                              self.lb[0].astype(np.float32) * box_b,
                              np.array([0], np.float32) * box_f))
        v_target = np.vstack((self.v0.astype(np.float32),
                              self.tb.astype(np.float32),
                              self.tb.astype(np.float32),
                              np.array([0], np.float32) * box_f))

        return np.hstack((x, t)), np.hstack((u_target, v_target))

    def __len__(self):
        return 1


def generate_PINNs_training_set(N0, Nb, Nf, lb, ub, path='./Data/NLS.mat'):
    """
    Generate training set for PINNs

    Args: see class PINNs_train_set
    """
    s = PINNs_training_set(N0, Nb, Nf, lb, ub, path)
    dataset = ds.GeneratorDataset(source=s, column_names=['data', 'label'], shuffle=False,
                                  python_multiprocessing=True)
    return dataset


def get_eval_data(path):
    """
    Get the evaluation data for Schrodinger equation.
    """
    data = scio.loadmat(path)
    t = data['tt'].astype(np.float32).flatten()[:, None]
    x = data['x'].astype(np.float32).flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact).astype(np.float32)
    Exact_v = np.imag(Exact).astype(np.float32)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    return X_star, u_star, v_star, h_star
