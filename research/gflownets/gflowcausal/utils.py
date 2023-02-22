# Copyright 2023 Huawei Technologies Co., Ltd
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
utils
"""
import warnings

import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LinearRegression, LassoLarsIC

from castle.datasets.simulator import IIDSimulation, DAG
from args import args


def save_sample_batch(reward_, tpr_, shd_, fdr_, p_tpr_, p_shd_, p_fdr_):
    df = pd.DataFrame(tpr_)
    df['shd'] = pd.DataFrame(shd_)
    df['fdr'] = pd.DataFrame(fdr_)
    df['p_tpr'] = pd.DataFrame(p_tpr_)
    df['p_shd'] = pd.DataFrame(p_shd_)
    df['p_fdr'] = pd.DataFrame(p_fdr_)
    df['reward'] = pd.DataFrame(reward_)

    df.columns = ['tpr', 'shd', 'fdr', 'p_tpr', 'p_shd', 'p_fdr', 'reward']

    df.to_csv('./sampling_results/batch_epo{}_nodes{}_edge{}_lr{}_hid{}_seed{}.csv'.format(
        args.epoches, args.n_node, args.n_edges, args.learning_rate, args.n_hid, args.seed))
    tpr_mean = np.mean(tpr_)
    shd_mean = np.mean(shd_)
    fdr_mean = np.mean(fdr_)
    p_tpr_mean = np.mean(p_tpr_)
    p_shd_mean = np.mean(p_shd_)
    p_fdr_mean = np.mean(p_fdr_)

    tpr_max = np.max(tpr_)
    print('\n tpr_mean {} , shd_mean {} , fdr_mean {} ,  p_tpr_mean {} , p_shd_mean {} ,p_fdr_mean {} '
          .format(tpr_mean, shd_mean, fdr_mean, p_tpr_mean, p_shd_mean, p_fdr_mean))
    print('\n tpr_max {} '.format(tpr_max))


def softmax_matrix(data):
    """
    The element in the matrix is greater than 0; and sum equals 1
    """
    data = softmax(data, axis=0)
    data_sum = np.sum(data)
    data /= data_sum
    return data


def select_action_base_probability(d_2, prob_data):
    results = []
    for i in range(args.mbsize):
        action_item = np.random.choice(np.arange(d_2), p=prob_data[i, :])
        results.append(action_item)
    return np.array(results)


def synthetic_data():
    weighted_random_dag = DAG.erdos_renyi(n_nodes=args.n_node, n_edges=args.n_edges, weight_range=(0.5, 2.0),
                                          seed=args.seed)
    datasets = IIDSimulation(weighted_random_dag, n=args.n_samples, method=args.sem_type, sem_type=args.reg_type)
    true_causal_matrix, X = datasets.B, datasets.X
    return true_causal_matrix, X


def synthetic_data_nonliear():
    weighted_random_dag = DAG.erdos_renyi(n_nodes=args.n_node, n_edges=args.n_edges, weight_range=(0.5, 2.0))
    datasets = IIDSimulation(weighted_random_dag, n=args.n_samples, method='nonlinear', sem_type=args.sem_type)
    true_causal_matrix, X = datasets.B, datasets.X
    return true_causal_matrix, X


def transitive_closure_update_ms(ori_transitive_matrix, update_row, update_column):
    """
    update transitive closure
    :param ori_transitive_matrix: origin transitive matrix
    :param update_row: num, update action，number of row in the state matrix
    :param update_column: num, update action，number of columns in the state matrix
    :return: updated transitive matrix
    """
    row_vec = ori_transitive_matrix[update_row, :].reshape(1, -1)
    col_vec = ori_transitive_matrix[:, update_column].reshape(-1, 1)
    outer_product = col_vec * row_vec
    updated_transitive_matrix = outer_product + ori_transitive_matrix
    return updated_transitive_matrix


def mask_update_ms(matrix, ori_transitive_matrix, update_row, update_column, order):
    """
    update mask matrix
    :param matrix: state matrix
    :param ori_transitive_matrix: origin transitive matrix
    :param update_row: num, update action，number of row in the state matrix
    :param update_column: num, update action，number of columns in the state matrix
    :param order: record order
    """
    updated_transitive_matrix = transitive_closure_update_ms(ori_transitive_matrix, update_row, update_column)
    updated_order = order_update_ms(order, update_row, update_column)
    update_order_mask = order_mask_update_ms(updated_order, d=matrix.shape[0])
    updated_mask = matrix + updated_transitive_matrix + update_order_mask
    return updated_mask, updated_transitive_matrix, updated_order


def order_update_ms(order, update_row, update_column):
    """
    update order
    :param order: array, record order
    :param update_row: num, update action，number of row in the state matrix
    :param update_column: num, update action，number of columns in the state matrix
    :return: updated order
    """
    update_row = np.array([update_row])
    update_column = np.array([update_column])
    len_order = len(order)
    if len_order == 0:
        updated_order = np.concatenate((order, update_column, update_row), axis=0)
    elif order[0] == update_row:  # out ——> in
        order_reverse = np.flip(order, axis=0)
        order_reverse = np.concatenate((order_reverse, update_column), axis=0)
        updated_order = np.flip(order_reverse, axis=0)
    else:  # in ——> out   order[-1] == update_column
        updated_order = np.concatenate((order, update_row), axis=0)
    return [updated_order]


def order_mask_update_ms(updated_order, d):
    """
    update order mask matrix
    """
    head_ix = int(updated_order[0][0])
    tail_ix = int(updated_order[0][-1])
    zeros_matrix = np.zeros(d * d).reshape(d, d)
    zeros_matrix[head_ix, :] = 1
    zeros_matrix[:, tail_ix] = 1
    order_mask = 1 - zeros_matrix
    return order_mask


class Reward:
    def __init__(self, cfg, X):
        self.d = {}  # store results
        self.seq_length = cfg.n_node
        self.n_samples = cfg.n_samples
        self.d_RSS = [{} for i in range(self.seq_length)]  # store RSS for reuse
        # self.regression_type = args.reg_type
        self.max_reward = 0
        self.regression_type = cfg.regression_type
        self.score_type = cfg.score_type
        self.input_data = X
        self.alpha = 1e-10

    def return_saved_results(self):
        return self.d

    def cal_ori(self, graph, order):
        graph_to_int2 = list(np.int32(order))
        graph_batch_to_tuple = tuple(graph_to_int2)
        if graph_batch_to_tuple in self.d:
            reward = self.d[graph_batch_to_tuple]
            return reward

        RSS_ls = []
        for i in range(self.seq_length):
            RSSi = self.cal_RSSi(i, graph)
            RSS_ls.append(RSSi)
        RSS_ls = np.array(RSS_ls)
        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls) / self.n_samples + 1e-8)
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls) / self.n_samples + 1e-8))
        reward = (1e+2) * np.exp(-BIC / 1e+2)
        self.d[graph_batch_to_tuple] = reward
        self.update_score(order=graph_batch_to_tuple, reward=reward)
        return reward

    def local_score_BIC(self, Data, i, PAi, parameters=None) -> float:
        if parameters is None:
            lambda_value = 1
        else:
            lambda_value = parameters['lambda_value']

        Data = np.mat(Data)
        T = Data.shape[0]
        X = Data[:, i]
        len_pai = len(PAi)
        if len_pai != 0:
            PA = Data[:, PAi]
            D = PA.shape[1]
            # derive the parameters by maximum likelihood
            PA = PA.squeeze(1)
            conduct = PA.T * PA
            H = PA * self.pdinv(conduct) * PA.T
            E = X - H * X
            sigma2 = np.sum(np.power(E, 2)) / T
            # BIC
            score = T * np.log(sigma2) + lambda_value * D * np.log(T)
        else:
            sigma2 = np.sum(np.power(X, 2)) / T
            # BIC
            score = T * np.log(sigma2)

        return score

    def pdinv(self, A):
        # PDINV Computes the inverse of a positive definite matrix
        numData = A.shape[0]
        try:
            U = np.linalg.cholesky(A).T
            invU = np.eye(numData).dot(np.linalg.inv(U))
            Ainv = invU.dot(invU.T)
        except np.linalg.LinAlgError as e:
            warnings.warn('Matrix is not positive definite in pdinv, inverting using svd')
            u, s, vh = np.linalg.svd(A, full_matrices=True)
            Ainv = vh.T.dot(np.diag(1 / s)).dot(u.T)
        except Exception as e:
            raise e
        return np.mat(Ainv)

    def update_score(self, order, reward):
        if self.max_reward <= reward:
            self.max_reward = reward
            self.max_rd_order = order

    def varsortability(self, X, order, tol=1e-9):
        """ Takes n x d data and a d x d adjaceny matrix,
        where the i,j-th entry corresponds to the edge weight for i->j,
        and returns a value indicating how well the variance order
        reflects the causal order. """
        graph_to_int2 = list(np.int32(order))
        graph_batch_to_tuple = tuple(graph_to_int2)
        if graph_batch_to_tuple in self.d:
            reward = self.d[graph_batch_to_tuple]
            return reward
        W = np.array(get_graph_from_order(sequence=order))

        E = W != 0
        Ek = E.copy()
        var = np.var(X, axis=0, keepdims=True)

        n_paths = 0
        n_correctly_ordered_paths = 0

        for _ in range(E.shape[0] - 1):
            n_paths += Ek.sum()
            n_correctly_ordered_paths += (Ek * var / var.T > 1 + tol).sum()
            n_correctly_ordered_paths += 1 / 2 * ((Ek * var / var.T <= 1 + tol) * (Ek * var / var.T > 1 - tol)).sum()
            Ek = Ek.dot(E)

        reward = 100 * n_correctly_ordered_paths / n_paths
        self.d[graph_batch_to_tuple] = reward
        self.update_score(order=graph_batch_to_tuple, reward=reward)
        return reward

    def best_result(self):
        return self.max_reward, self.max_rd_order


def pruning_by_sortnregress(order, graph_batch, X, thresh=args.thresh):
    order_list = list(order)
    W_sort = sortnregress(X, order_list)

    W = np.multiply(graph_batch, W_sort)
    return np.float32(np.abs(W) > thresh)


def sortnregress(X, order_list):
    """ Take n x d data, order nodes by marginal variance and
    regresses each node onto those with lower variance, using
    edge coefficients as structure estimates. """
    LR = LinearRegression()
    model = LassoLarsIC(criterion='bic', normalize=False)

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))

    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]
        LR.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        X_new = X[:, covariates] * weight
        y_new = X[:, target].ravel()
        model.fit(X_new, y_new)
        W[covariates, target] = model.coef_ * weight

    return W


def pruning_by_coef(graph_batch, X, thresh=args.thresh) -> np.ndarray:
    """
    for a given graph, pruning the edge according to edge weights;
    linear regression for each causal regression for edge weights and
    then thresholding
    """
    n, d = X.shape
    reg = LinearRegression()
    W = []
    loss = 0
    for i in range(d):
        col = np.abs(graph_batch[i]) > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue
        X_train = X[:, col]
        y = X[:, i]
        reg.fit(X_train, y)
        loss += 0.5 / n * np.sum(np.square(reg.predict(X_train) - y))
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d,)
        for ci in range(d):
            if col[ci]:
                new_reg_coeff[ci] = reg_coeff[cj]
                cj += 1

        W.append(new_reg_coeff)
    return np.float32(np.abs(W) > thresh)


def get_graph_from_order(sequence, dag_mask=None) -> np.ndarray:
    """
    Generate a fully-connected DAG based on a sequence.

    Parameters
    ----------
    sequence: iterable
        An ordering of nodes, the set of nodes that precede node vj
        denotes potential parent nodes of vj.
    dag_mask : ndarray
        two-dimensional array with [0, 1], shape = [n_nodes, n_nodes].
        (i, j) indicated element `0` denotes there must be no edge
        between nodes `i` and `j` , the element `1` indicates that
        there may or may not be an edge.

    Returns
    -------
    out:
        graph matrix

    Examples
    --------
    >>> order = [2, 0, 1, 3]
    >>> graph = get_graph_from_order(sequence=order)
    >>> print(graph)
        [[0. 1. 0. 1.]
         [0. 0. 0. 1.]
         [1. 1. 0. 1.]
         [0. 0. 0. 0.]]
    """

    num_node = len(sequence)
    init_graph = np.zeros((num_node, num_node))
    for i in range(num_node - 1):
        pa_node = sequence[i]
        sub_node = sequence[i + 1:]
        init_graph[pa_node, sub_node] = 1
    if dag_mask is None:
        gtrue_mask = np.ones([num_node, num_node]) - np.eye(num_node)
    else:
        gtrue_mask = dag_mask
    dag_mask = np.int32(np.abs(gtrue_mask) > 1e-3)
    init_graph = init_graph * dag_mask

    return init_graph
