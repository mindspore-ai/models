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
loss function script.
"""
import heapq
import numpy as np


def calculate_loss(inp, target, classes, n_support, n_query, n_class, is_train=True):
    """
    loss construct
    """
    n_classes = len(classes)
    support_idxs = ()
    query_idxs = ()

    for ind, _ in enumerate(classes):
        class_c = classes[ind]
        matrix = np.equal(target, class_c).astype(np.float32)
        K = n_support + n_query
        a = heapq.nlargest(K, range(len(matrix)), matrix.take)
        support_idx = np.squeeze(a[:n_support])
        support_idxs += (support_idx,)
        query_idx = a[n_support:]
        query_idxs += (query_idx,)

    prototypes = ()
    for idx_list in support_idxs:
        prototypes += (np.mean(inp[idx_list], axis=0),)
    prototypes = np.stack(prototypes)
    query_idxs = np.stack(query_idxs).reshape(-1)
    query_samples = inp[query_idxs]


    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = np.log(np.exp(-dists) / np.sum(np.exp(-dists)))
    log_p_y = log_p_y.reshape((n_classes, n_query, -1))

    target_inds = np.arange(0, n_class, dtype=np.int32).reshape((n_classes, 1, 1))
    target_inds = np.broadcast_to(target_inds, (n_classes, n_query, 1))
    loss_val = -np.mean(np.squeeze(gather(log_p_y, 2, target_inds).reshape(-1)))

    y_hat = np.argmax(log_p_y, axis=2)
    acc_val = np.mean(np.equal(y_hat, np.squeeze(target_inds)).astype(np.float32))
    if is_train:
        return loss_val
    return acc_val, loss_val

def supp_idxs(target, c):
    return np.squeeze(nonZero(np.equal(target, c))[:n_support])

def nonZero(inpbool):
    out = []
    for _, inp in enumerate(inpbool):
        if inp:
            out.append(inp)
    return np.array(out, dtype=np.int32)

def acc():
    return acc_val

def gather(self, dim, index):
    '''
    gather
    '''
    idx_xsection_shape = index.shape[:dim] + \
        index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    x = np.broadcast_to(np.expand_dims(x, axis=1), (n, m, d))
    y = np.broadcast_to(np.expand_dims(y, axis=0), (n, m, d))

    return np.sum(np.power(x - y, 2), 2)
