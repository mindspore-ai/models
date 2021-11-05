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
Probability distributions.
"""
import math
import numpy as np

import mindspore.nn as nn
from mindspore.ops import operations as ops

PI = math.pi

_exp = ops.Exp()
_sum = ops.ReduceSum()
_log = ops.Log()
_logsigmoid = nn.LogSigmoid()
_sigmoid = ops.Sigmoid()
_erf = ops.Erf()


def log_min_exp(a, b, epsilon=1e-8):
    """log-min-exp operation"""
    # assume b < a
    y = a + _log(1 - _exp(b - a) + epsilon)
    return y


def log_normal(x, mean, logvar):
    """log probability of normal distribution"""
    logp = -0.5 * logvar
    logp += -0.5 * np.log(2 * PI)
    logp += -0.5 * (x - mean) * (x - mean) / _exp(logvar)
    return logp


def log_mixture_normal(x, mean, logvar, pi):
    """log probability of mixture of normals"""
    x = x.view(*x.shape, 1)
    logp_mixtures = log_normal(x, mean, logvar)
    logp = _log(_sum(pi * _exp(logp_mixtures), axis=-1) + 1e-8)
    return logp


def log_logistic(x, mean, logscale):
    """log probability of logistic distribution"""
    scale = _exp(logscale)
    u = (x - mean) / scale
    logp = _logsigmoid(u) + _logsigmoid(-u) - logscale
    return logp


def log_discretized_logistic(x, mean, logscale, inverse_bin_width):
    """log density of discretized logistic distribution"""
    scale = _exp(logscale)

    logp = log_min_exp(
        _logsigmoid((x + 0.5 / inverse_bin_width - mean) / scale),
        _logsigmoid((x - 0.5 / inverse_bin_width - mean) / scale))

    return logp


def normal_cdf(value, loc, std):
    """CDF of normal distribution"""
    return 0.5 * (1 + _erf((value - loc) / std / math.sqrt(2)))


def log_discretized_normal(x, mean, logvar, inverse_bin_width):
    """log density of discretized normal distribution"""
    std = _exp(0.5 * logvar)
    log_p = _log(normal_cdf(x + 0.5 / inverse_bin_width, mean, std)
                 - normal_cdf(x - 0.5 / inverse_bin_width, mean, std) + 1e-7)
    return log_p


def log_mixture_discretized_normal(x, mean, logvar, pi, inverse_bin_width):
    """log density of discretized mixture of normals"""
    std = _exp(0.5 * logvar)
    x = x.view(*x.shape, 1)
    p = normal_cdf(x + 0.5 / inverse_bin_width, mean, std) \
        - normal_cdf(x - 0.5 / inverse_bin_width, mean, std)
    p = _sum(p * pi, axis=-1)
    logp = _log(p + 1e-8)
    return logp


def log_mixture_discretized_logistic(x, mean, logscale, pi, inverse_bin_width):
    """log density of discretized mixture of logistics"""
    scale = _exp(logscale)
    x = x.view(*x.shape, 1)
    p = _sigmoid((x + 0.5 / inverse_bin_width - mean) / scale) \
        - _sigmoid((x - 0.5 / inverse_bin_width - mean) / scale)
    p = _sum(p * pi, axis=-1)
    logp = _log(p + 1e-8)
    return logp


def mixture_discretized_logistic_cdf(x, mean, logscale, pi, inverse_bin_width):
    """CDF of discretized mixture of logistics"""
    scale = _exp(logscale)
    x = x.view(*x.shape, 1)
    cdfs = _sigmoid((x + 0.5 / inverse_bin_width - mean) / scale)
    cdf = _sum(cdfs * pi, axis=-1)
    return cdf


def mixture_discretized_normal_cdf(x, mean, logvar, pi, inverse_bin_width):
    """CDF of discretized mixture of normals"""
    std = _exp(0.5 * logvar)
    x = x.view(*x.shape, 1)
    cdfs = normal_cdf(x + 0.5 / inverse_bin_width, mean, std)
    cdf = _sum(cdfs * pi, axis=-1)
    return cdf
