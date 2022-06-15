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

"""generate diffusion and ppmi matrix"""
import random
import itertools
import math
import scipy
import scipy.sparse as sp
import numpy as np
from numpy import inf


def diffusion_fun_sparse(A):
    """Function of Calculating diffusion matrix."""
    n, m = A.shape
    A_with_selfloop = A + sp.identity(n, format='csc')
    diags = A_with_selfloop.sum(axis=1).flatten()

    with scipy.errstate(divide='ignore'):# Context manager for error handling, ignoring exceptions divided by zero
        diags_sqrt = 1.0 / scipy.sqrt(diags)
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], m, n, format='csc')
    d = DH.dot(A_with_selfloop.dot(DH))
    return d


def _normalize_diffusion_matrix(A):
    """Used to nomalized diffusion matrix."""
    n, m = A.shape
    A_with_selfloop = A
    diags = A_with_selfloop.sum(axis=1).flatten()

    with scipy.errstate(divide='ignore'):
        diags_sqrt = 1.0 / scipy.sqrt(diags)
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], m, n, format='csc')
    d = DH.dot(A_with_selfloop.dot(DH))
    return d


def diffusion_fun_improved(A, sampling_num=100, path_len=3,
                           self_loop=True, spars=False):
    """Return normalized adjacent matrix plus PPMI."""
    print("Do the sampling...")
    mat = _diffusion_fun_sampling(
        A, sampling_num=sampling_num, path_len=path_len,
        self_loop=self_loop, spars=spars)
    print("Calculating the PPMI...")
    # mat is a sparse lil_matrix
    pmi = None
    if spars:
        pmi = _PPMI_sparse(mat)
    else:
        pmi = _PPMI(mat)
    A_with_selfloop = A + pmi
    dig = np.sum(A_with_selfloop, axis=1)
    dig = np.squeeze(np.asarray(dig))
    Degree = np.diag(dig)
    Degree_normalized = Degree ** (-0.5)
    Degree_normalized[Degree_normalized == inf] = 0.0
    Diffusion = np.dot(
        np.dot(Degree_normalized, A_with_selfloop), Degree_normalized)
    return Diffusion


def diffusion_fun_improved_ppmi_dynamic_sparsity(A, sampling_num=100, path_len=2,
                                                 self_loop=True, spars=True, k=1.0):
    """Do the sampling and calculate the PPMI."""
    print("Do the sampling...")
    mat = _diffusion_fun_sampling(
        A, sampling_num=sampling_num, path_len=path_len,
        self_loop=self_loop, spars=spars)
    print("Calculating the PPMI...")
    # mat is a sparse dok_matrix
    if spars:
        pmi = _PPMI_sparse(mat)
    else:
        pmi = _PPMI(mat)

    pmi = _shift(pmi, k)
    ans = _normalize_diffusion_matrix(pmi.tocsc())

    return ans


def _shift(mat, k):
    """Shift function."""
    print(k)
    r, c = mat.shape
    x, y = mat.nonzero()
    mat = mat.todok()
    offset = np.log(k)
    print("Offset: " + str(offset))
    for i, j in zip(x, y):
        mat[i, j] = max(mat[i, j] - offset, 0)

    x, y = mat.nonzero()
    sparsity = 1.0 - len(x) / float(r * c)
    print("Sparsity: " + str(sparsity))
    return mat


def _diffusion_fun_sampling(A, sampling_num=100, path_len=3, self_loop=True, spars=False):
    """Return diffusion matrix."""
    re = None
    if not spars:
        re = np.zeros(A.shape)
    else:
        re = sp.dok_matrix(A.shape, dtype=np.float32)

    if self_loop:
        A_with_selfloop = A + sp.identity(A.shape[0], format="csr")#A+I
    else:
        A_with_selfloop = A

    # record each node's neignbors
    dict_nid_neighbors = {}
    for nid in range(A.shape[0]):
        neighbors = np.nonzero(A_with_selfloop[nid])[1]
        dict_nid_neighbors[nid] = neighbors
    # for each node
    for i in range(A.shape[0]):
        # for each sampling iter
        for dummy_j in range(sampling_num):
            _generate_path(i, dict_nid_neighbors, re, path_len)
    return re


def _generate_path(node_id, dict_nid_neighbors, re, path_len):
    """generate path of each node"""
    path_node_list = [node_id]
    for dummy_i in range(path_len - 1):
        temp = dict_nid_neighbors.get(path_node_list[-1])
        if temp.all() == 0:
            break
        else:
            path_node_list.append(random.choice(temp))

    # update difussion matrix re
    for pair in itertools.combinations(path_node_list, 2):
        if pair[0] == pair[1]:
            re[pair[0], pair[1]] += 1.0
        else:
            re[pair[0], pair[1]] += 1.0
            re[pair[1], pair[0]] += 1.0


def _PPMI(mat):
    """generate PPMI matrix"""
    (nrows, ncols) = mat.shape
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    # print rowTotals.shape
    N = np.sum(rowTotals)
    rowMat = np.ones((nrows, ncols), dtype=np.float32)
    for i in range(nrows):
        rowMat[i, :] = 0 if rowTotals[i] == 0 else rowMat[i, :] * (1.0 / rowTotals[i])
    colMat = np.ones((nrows, ncols), dtype=np.float)
    for j in range(ncols):
        colMat[:, j] = 0 if colTotals[j] == 0 else colMat[:, j] * (1.0 / colTotals[j])
    P = N * mat * rowMat * colMat
    P = np.fmax(np.zeros((nrows, ncols), dtype=np.float32), np.log(P))
    return P


def _PPMI_sparse(mat):
    """generate sparse PPMI matrix"""
    # mat is a sparse dok_matrix
    nrows, ncols = mat.shape
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T

    N = float(np.sum(rowTotals))
    rows, cols = mat.nonzero()

    p = sp.dok_matrix((nrows, ncols))
    for i, j in zip(rows, cols):
        _under = rowTotals[0, i] * colTotals[0, j]
        if _under != 0.0:
            log_r = np.log((N * mat[i, j]) / _under)
            if log_r > 0:
                p[i, j] = log_r
    return p


def rampup(epoch, scaled_unsup_weight_max, exp=5.0, rampup_length=80):
    """Use to calculate the coefficient of ppmiloss"""
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        ramp = math.exp(-p * p * exp) * scaled_unsup_weight_max
    else:
        ramp = 1.0 * scaled_unsup_weight_max
    return ramp


def get_scaled_unsup_weight_max(num_labels, X_train_shape, unsup_weight_max=100.0):
    """Use to calculate the coefficient of ppmiloss"""
    return unsup_weight_max * 1.0 * num_labels / X_train_shape
