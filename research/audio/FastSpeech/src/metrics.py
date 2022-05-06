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
"""Metrics scripts."""
import numpy as np


def kernel_classifier_distance_and_std_from_activations(
        activations1,
        activations2,
        max_block_size=1024,
        dtype=np.float32,
):
    """Compute kernel distance between two activations."""
    n_r = activations1.shape[0]
    n_g = activations2.shape[0]

    n_bigger = np.maximum(n_r, n_g)
    n_blocks = np.ceil(n_bigger / max_block_size).astype(np.int32)

    v_r = n_r // n_blocks
    v_g = n_g // n_blocks

    n_plusone_r = n_r - v_r * n_blocks
    n_plusone_g = n_g - v_g * n_blocks

    sizes_r = np.concatenate([np.full([n_blocks - n_plusone_r], v_r), np.full([n_plusone_r], v_r + 1)], axis=0)

    sizes_g = np.concatenate([
        np.full([n_blocks - n_plusone_g], v_g),
        np.full([n_plusone_g], v_g + 1)], axis=0)

    zero = np.zeros([1], dtype=np.int32)
    inds_r = np.concatenate([zero, np.cumsum(sizes_r)], axis=0)
    inds_g = np.concatenate([zero, np.cumsum(sizes_g)], axis=0)

    dim = activations1.shape[1]

    def compute_kid_block(i):
        """Computes the ith block of the KID estimate."""
        r_s = inds_r[i]
        r_e = inds_r[i + 1]
        r = activations1[r_s:r_e]
        m = (r_e - r_s).astype(dtype)

        g_s = inds_g[i]
        g_e = inds_g[i + 1]
        g = activations2[g_s:g_e]
        n = (g_e - g_s).astype(dtype)

        k_rr = (np.matmul(r, r.T) / dim + 1) ** 3
        k_rg = (np.matmul(r, g.T) / dim + 1) ** 3
        k_gg = (np.matmul(g, g.T) / dim + 1) ** 3

        out = (-2 * np.mean(k_rg) + (np.sum(k_rr) - np.trace(k_rr)) /
               (m * (m - 1)) + (np.sum(k_gg) - np.trace(k_gg)) / (n * (n - 1)))

        return out.astype(dtype)

    ests = np.array([compute_kid_block(i) for i in range(n_blocks)])

    mn = np.mean(ests)

    n_blocks_ = n_blocks.astype(dtype)

    if np.less_equal(n_blocks, 1):
        var = np.array(float('nan'), dtype=dtype)
    else:
        var = np.sum(np.square(ests - mn)) / (n_blocks_ - 1)

    return mn, np.sqrt(var / n_blocks_)


def frechet_classifier_distance_from_activations(
        activations1,
        activations2,
):
    """Compute frechet distance between two activations."""
    activations1 = activations1.astype(np.float64)
    activations2 = activations2.astype(np.float64)

    m = np.mean(activations1, axis=0)
    m_w = np.mean(activations2, axis=0)

    # Calculate the unbiased covariance matrix of first activations.
    num_examples_real = activations1.shape[0]
    sigma = num_examples_real / (num_examples_real - 1) * np.cov(activations1.T)
    # Calculate the unbiased covariance matrix of second activations.
    num_examples_generated = activations2.shape[0]
    sigma_w = num_examples_generated / (num_examples_generated - 1) * np.cov(activations2.T)

    def _calculate_fid(m, m_w, sigma, sigma_w):
        """Returns the Frechet distance given the sample mean and covariance."""
        # Find the Tr(sqrt(sigma sigma_w)) component of FID
        sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

        # Compute the two components of FID.

        # First the covariance component.
        # Here, note that trace(A + B) = trace(A) + trace(B)
        trace = np.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

        # Next the distance between means.
        mean = np.sum(squared_difference(m, m_w))

        # Equivalent to L2 but more stable.
        fid = trace + mean

        return fid.astype(np.float64)

    result = tuple(
        _calculate_fid(m_val, m_w_val, sigma_val, sigma_w_val) for
        m_val, m_w_val, sigma_val, sigma_w_val in
        zip([m], [m_w], [sigma], [sigma_w])
    )

    return result[0]


def squared_difference(m, w):
    arr = []
    for i, j in zip(m, w):
        arr.append((i - j) ** 2)
    arr = np.array(arr)

    return arr


def trace_sqrt_product(sigma, sigma_v):
    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = _symmetric_matrix_square_root(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = np.matmul(sqrt_sigma, np.matmul(sigma_v, sqrt_sigma))

    return np.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = np.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = np.where(np.less(s, eps), s, np.sqrt(s))

    return np.matmul(np.matmul(u, np.diag(si)), v)
