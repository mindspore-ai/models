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
"Modify from https://github.com/goodfeli/adversarial"

import gc
from multiprocessing import Pool
from functools import partial
import numpy as np

def parzen_estimation(mu, sigma, mode='gauss'):
    """
    Implementation of a parzen-window estimation
    Keyword arguments:
        x: A "nxd"-dimentional numpy array, which each sample is
                  stored in a separate row (=training example)
        mu: point x for density estimation, "dx1"-dimensional numpy array
        sigma: window width
    Return the density estimate p(x)
    """
    def log_mean_exp(a):
        max_ = a.max(axis=1)
        return max_ + np.log(np.exp(a - np.expand_dims(max_, axis=1)).mean(1))

    def gaussian_window(x, mu, sigma):
        a = (np.expand_dims(x, axis=1) - np.expand_dims(mu, axis=0)) / sigma
        E = log_mean_exp(-0.5 * (a ** 2).sum(-1))
        Z = mu.shape[1] * np.log(sigma * np.sqrt(np.pi * 2))
        return E - Z

    def hypercube_kernel(x, mu, h):
        n, d = mu.shape
        a = (np.expand_dims(x, axis=1) - np.expand_dims(mu, axis=0)) / h
        b = np.all(np.less(np.abs(a), 1/2), axis=-1)
        kn = np.sum(b.astype(int), axis=-1)
        return kn / (n * h**d)

    if mode == 'gauss':
        return lambda x: gaussian_window(x, mu, sigma)
    return lambda x: hypercube_kernel(x, mu, h=sigma)


# Pdf estimation
def pdf_multivaraible_gauss(x, mu, cov):
    part1 = 1 / ((2 * np.pi) ** (len(mu) / 2) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * (x - mu).T.dot(np.linalg.inv(cov)).dot((x - mu))
    return part1 * np.exp(part2)


def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))
    nlls = []
    for i in range(n_batches):
        nll = parzen(x[inds[i::n_batches]])
        nlls.extend(nll)
    return np.array(nlls)

def find_sigma(samples, data, batch_size, sigma):
    parzen = parzen_estimation(samples, sigma, mode='gauss')
    ll = get_nll(data, parzen, batch_size=batch_size)
    del parzen
    gc.collect()
    print('sigma:', sigma, 'll_m:', ll.mean())
    return ll.mean()

def cross_validate_sigma(samples, data, sigmas, batch_size, num_of_thread):

    lls = []
    p = Pool(num_of_thread)
    func = partial(find_sigma, samples, data, batch_size)
    lls = p.map(func, sigmas)
    ind = np.argmax(lls)
    return sigmas[ind], lls
