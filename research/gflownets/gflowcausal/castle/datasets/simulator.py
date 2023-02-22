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
import logging
import random
from random import sample
from copy import deepcopy
from itertools import combinations

import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from scipy.special import expit as sigmoid


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class DAG:
    '''
    A class for simulating random (causal) DAG, where any DAG generator
    method would return the weighed/binary adjacency matrix of a DAG.
    Besides, we recommend using the python package "NetworkX"
    to create more structures types.
    '''

    @staticmethod
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    @staticmethod
    def _random_acyclic_orientation(B_und):
        B = np.tril(DAG._random_permutation(B_und), k=-1)
        B_perm = DAG._random_permutation(B)
        return B_perm

    @staticmethod
    def _graph_to_adjmat(G):
        return nx.to_numpy_matrix(G)

    @staticmethod
    def _BtoW(B, d, w_range):
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        W = (B != 0).astype(float) * U
        return W

    @staticmethod
    def _low_rank_dag(d, degree, rank):
        """
        Simulate random low rank DAG with some expected degree.

        Parameters
        ----------
        d: int
            Number of nodes.
        degree: int
            Expected node degree, in + out.
        rank: int
            Maximum rank (rank < d-1).

        Return
        ------
        B: np.nparray
            Initialize DAG.
        """
        prob = float(degree) / (d - 1)
        B = np.triu((np.random.rand(d, d) < prob).astype(float), k=1)
        total_edge_num = np.sum(B == 1)
        sampled_pa = sample(range(d - 1), rank)
        sampled_pa.sort(reverse=True)
        sampled_ch = []
        for i in sampled_pa:
            candidate = set(range(i + 1, d))
            candidate = candidate - set(sampled_ch)
            sampled_ch.append(sample(candidate, 1)[0])
            B[i, sampled_ch[-1]] = 1
        remaining_pa = list(set(range(d)) - set(sampled_pa))
        remaining_ch = list(set(range(d)) - set(sampled_ch))
        B[np.ix_(remaining_pa, remaining_ch)] = 0
        # mask B
        maskedB = B + np.tril(np.ones((d, d)))
        maskedB[np.ix_(remaining_pa, remaining_ch)] = 1
        B[maskedB == 0] = 1
        remaining_ch = [i + d for i in remaining_ch]
        remaining_ch_set = set(remaining_ch)
        remaining_pa_set = set(remaining_pa)

        edges = np.transpose(np.nonzero(B))
        edges[:, 1] += d
        bigraph = nx.Graph()
        bigraph.add_nodes_from(range(2 * d))
        bigraph.add_edges_from(edges)
        M = nx.bipartite.maximum_matching(bigraph, top_nodes=range(d))
        while len(M) > 2 * rank:
            keys = set(M.keys())
            rmv_cand = keys & (remaining_pa_set | remaining_ch_set)
            p = sample(rmv_cand, 1)[0]
            c = M[p]
            # destroy p-c
            bigraph.remove_edge(p, c)
            M = nx.bipartite.maximum_matching(bigraph, top_nodes=range(d))

        new_edges = np.array(bigraph.edges)
        for i in range(len(new_edges)):
            new_edges[i,].sort()
        new_edges[:, 1] -= d

        BB = np.zeros((d, d))
        B = np.zeros((d, d))
        BB[new_edges[:, 0], new_edges[:, 1]] = 1

        if np.sum(BB == 1) > total_edge_num:
            delta = total_edge_num - rank
            BB[sampled_pa, sampled_ch] = 0
            rmv_cand_edges = np.transpose(np.nonzero(BB))
            if delta <= 0:
                raise RuntimeError(r'Number of edges is below the rank, please \
                                   set a larger edge or degree \
                                   (you can change seed or increase degree).')
            selected = np.array(sample(rmv_cand_edges.tolist(), delta))
            B[selected[:, 0], selected[:, 1]] = 1
            B[sampled_pa, sampled_ch] = 1
        else:
            B = deepcopy(BB)

        B = B.transpose()
        return B

    @staticmethod
    def erdos_renyi(n_nodes, n_edges, weight_range=None, seed=None):
        assert n_nodes > 0
        set_random_seed(seed)
        # Erdos-Renyi
        creation_prob = (2 * n_edges) / (n_nodes ** 2)
        G_und = nx.erdos_renyi_graph(n=n_nodes, p=creation_prob, seed=seed)
        B_und = DAG._graph_to_adjmat(G_und)
        B = DAG._random_acyclic_orientation(B_und)
        if weight_range is None:
            return B
        W = DAG._BtoW(B, n_nodes, weight_range)
        return W


    @staticmethod
    def bipartite(n_nodes, n_edges, split_ratio=0.2, weight_range=None, seed=None):
        assert n_nodes > 0
        set_random_seed(seed)
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        n_top = int(split_ratio * n_nodes)
        n_bottom = n_nodes - n_top
        creation_prob = n_edges / (n_top * n_bottom)
        G_und = bipartite.random_graph(n_top, n_bottom, p=creation_prob, directed=True)
        B_und = DAG._graph_to_adjmat(G_und)
        B = DAG._random_acyclic_orientation(B_und)
        if weight_range is None:
            return B
        W = DAG._BtoW(B, n_nodes, weight_range)
        return W

    @staticmethod
    def hierarchical(n_nodes, degree=5, graph_level=5, weight_range=None, seed=None):
        assert n_nodes > 1
        set_random_seed(seed)
        prob = float(degree) / (n_nodes - 1)
        B = np.tril((np.random.rand(n_nodes, n_nodes) < prob).astype(float), k=-1)
        point = sample(range(n_nodes - 1), graph_level - 1)
        point.sort()
        point = [0] + [x + 1 for x in point] + [n_nodes]
        for i in range(graph_level):
            B[point[i]:point[i + 1], point[i]:point[i + 1]] = 0
        if weight_range is None:
            return B
        W = DAG._BtoW(B, n_nodes, weight_range)
        return W

    @staticmethod
    def low_rank(n_nodes, degree=1, rank=5, weight_range=None, seed=None):

        assert n_nodes > 0
        set_random_seed(seed)
        B = DAG._low_rank_dag(n_nodes, degree, rank)
        if weight_range is None:
            return B
        W = DAG._BtoW(B, n_nodes, weight_range)
        return W


class IIDSimulation:
    '''
    Simulate IID datasets for causal structure learning.

    Parameters
    ----------
    W: np.ndarray
        Weighted adjacency matrix for the target causal graph.
    n: int
        Number of samples for standard trainning dataset.
    method: str, (linear or nonlinear), default='linear'
        Distribution for standard trainning dataset.
    sem_type: str
        gauss, exp, gumbel, uniform, logistic (linear);
        mlp, mim, gp, gp-add, quadratic (nonlinear).
    noise_scale: float
        Scale parameter of noise distribution in linear SEM.
    '''

    def __init__(self, W, n=1000, method='linear',
                 sem_type='gauss', noise_scale=1.0):

        self.B = (W != 0).astype(int)
        if method == 'linear':
            self.X = IIDSimulation._simulate_linear_sem(
                W, n, sem_type, noise_scale)
        elif method == 'nonlinear':
            print('nonlinear!')
            self.X = IIDSimulation._simulate_nonlinear_sem(
                W, n, sem_type, noise_scale)
        logging.info('Finished synthetic dataset')

    @staticmethod
    def _simulate_linear_sem(W, n, sem_type, noise_scale):
        """
        Simulate samples from linear SEM with specified type of noise.
        For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

        Parameters
        ----------
        W: np.ndarray
            [d, d] weighted adj matrix of DAG.
        n: int
            Number of samples, n=inf mimics population risk.
        sem_type: str
            gauss, exp, gumbel, uniform, logistic.
        noise_scale: float
            Scale parameter of noise distribution in linear SEM.

        Return
        ------
        X: np.ndarray
            [n, d] sample matrix, [d, d] if n=inf
        """

        def _simulate_single_equation(X, w, scale):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            if sem_type == 'gauss':
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'exp':
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            elif sem_type == 'logistic':
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            else:
                raise ValueError('Unknown sem type. In a linear model, \
                                 the options are as follows: gauss, exp, \
                                 gumbel, uniform, logistic.')
            return x

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        G_nx = nx.from_numpy_matrix(W, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(G_nx):
            raise ValueError('W must be a DAG')
        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type == 'gauss':
                # make 1/d X'X = true cov
                X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
                return X
            raise ValueError('population risk not available')
        # empirical risk
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        return X

    @staticmethod
    def _simulate_nonlinear_sem(W, n, sem_type, noise_scale):
        """
        Simulate samples from nonlinear SEM.

        Parameters
        ----------
        B: np.ndarray
            [d, d] binary adj matrix of DAG.
        n: int
            Number of samples.
        sem_type: str
            mlp, mim, gp, gp-add, or quadratic.
        noise_scale: float
            Scale parameter of noise distribution in linear SEM.

        Return
        ------
        X: np.ndarray
            [n, d] sample matrix
        """
        if sem_type == 'quadratic':
            print('quadratic')
            return IIDSimulation._simulate_quad_sem(W, n, noise_scale)

        def _simulate_single_equation(X, scale):
            """X: [n, num of parents], x: [n]"""
            z = np.random.normal(scale=scale, size=n)
            pa_size = X.shape[1]
            if pa_size == 0:
                return z
            if sem_type == 'mlp':
                hidden = 100
                W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                x = sigmoid(X @ W1) @ W2 + z
            elif sem_type == 'mim':
                w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w1[np.random.rand(pa_size) < 0.5] *= -1
                w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w2[np.random.rand(pa_size) < 0.5] *= -1
                w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w3[np.random.rand(pa_size) < 0.5] *= -1
                x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            elif sem_type == 'gp-add':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                         for i in range(X.shape[1])]) + z
            else:
                raise ValueError('Unknown sem type. In a nonlinear model, \
                                 the options are as follows: mlp, mim, \
                                 gp, gp-add, or quadratic.')
            return x

        B = (W != 0).astype(int)
        d = B.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale

        X = np.zeros([n, d])
        G_nx = nx.from_numpy_matrix(B, create_using=nx.DiGraph)
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
        return X

    @staticmethod
    def _simulate_quad_sem(W, n, noise_scale):
        """
        Simulate samples from SEM with specified type of noise.
        Coefficient is randomly drawn but specifically designed
        to avoid overflow issues.

        Parameters
        ----------
        W: np.ndarray
            weigted DAG.
        n: int
            Number of samples.
        noise_scale: float
            Scale parameter of noise distribution in linear SEM.

        Return
        ------
        X: np.ndarray
            [n,d] sample matrix
        """

        def generate_quadratic_coef(random_zero=True):
            if random_zero and np.random.randint(low=0, high=2):
                return 0
            coef = np.random.uniform(low=0.5, high=1)
            if np.random.randint(low=0, high=2):
                coef *= -1
            return coef

        G = nx.DiGraph(W)
        d = W.shape[0]
        X = np.zeros([n, d])
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            len_p = len(parents)
            if len_p == 0:
                eta = np.zeros([n])
            elif len_p == 1:
                # We don't generate random zero coefficient if there is only one parent
                eta = np.zeros([n])
                used_parents = set()
                p = parents[0]
                num_terms = 0

                # Linear term
                coef = generate_quadratic_coef(random_zero=False)
                if coef != 0:
                    eta += coef * X[:, p]
                    used_parents.add(p)
                    num_terms += 1

                # Squared term
                coef = generate_quadratic_coef(random_zero=False)
                if coef != 0:
                    eta += coef * np.square(X[:, p])
                    used_parents.add(p)
                    num_terms += 1

                if num_terms > 0:
                    eta /= num_terms  # Compute average

                # Remove parent if both coef is zero
                if p not in used_parents:
                    W[p, j] = 0
            else:  # More than 1 parent
                eta = np.zeros([n])
                used_parents = set()
                num_terms = 0

                for p in parents:
                    # Linear terms
                    coef = generate_quadratic_coef(random_zero=True)
                    if coef > 0:
                        eta += coef * X[:, p]
                        used_parents.add(p)
                        num_terms += 1

                    # Squared terms
                    coef = generate_quadratic_coef(random_zero=True)
                    if coef > 0:
                        eta += coef * np.square(X[:, p])
                        used_parents.add(p)
                        num_terms += 1

                # Cross terms
                for p1, p2 in combinations(parents, 2):
                    coef = generate_quadratic_coef(random_zero=True)
                    if coef > 0:
                        eta += coef * X[:, p1] * X[:, p2]
                        used_parents.add(p1)
                        used_parents.add(p2)
                        num_terms += 1

                if num_terms > 0:
                    eta /= num_terms  # Compute average

                # Remove parent if both coef is zero
                unused_parents = set(parents) - used_parents
                if parents[-1] in unused_parents:
                    W[parents[-1], j] = 0

            X[:, j] = eta + np.random.normal(scale=noise_scale, size=n)

        return X
