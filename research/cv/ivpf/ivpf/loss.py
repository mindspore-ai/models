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
Loss functions.
"""
import numpy as np

from mindspore.ops import operations as ops

from .distributions import log_discretized_logistic, \
    log_discretized_normal, log_logistic, log_normal, \
    log_mixture_discretized_logistic, log_mixture_discretized_normal, \
    log_mixture_normal


def compute_log_ps(pxs, xs, args):
    """get log probability of different distributions"""
    # Add likelihoods of intermediate representations.
    inverse_bin_width = 2.**args.n_bits

    log_pxs = []
    for px, x in zip(pxs, xs):

        if args.variable_type == 'discrete':
            if args.distribution_type == 'logistic':
                log_px = log_discretized_logistic(
                    x, *px, inverse_bin_width=inverse_bin_width)
            elif args.distribution_type == 'normal':
                log_px = log_discretized_normal(
                    x, *px, inverse_bin_width=inverse_bin_width)
        elif args.variable_type == 'continuous':
            if args.distribution_type == 'logistic':
                log_px = log_logistic(x, *px)
            elif args.distribution_type == 'normal':
                log_px = log_normal(x, *px)

        log_pxs.append(ops.ReduceSum()(log_px, axis=[1, 2, 3]))

    return log_pxs


def compute_log_pz(pz, z, args):
    """get log probability of prior of different distributions"""
    inverse_bin_width = 2.**args.n_bits

    if args.variable_type == 'discrete':
        if args.distribution_type == 'logistic':
            if args.n_mixtures == 1:
                log_pz = log_discretized_logistic(
                    z, pz[0], pz[1], inverse_bin_width=inverse_bin_width)
            else:
                log_pz = log_mixture_discretized_logistic(
                    z, pz[0], pz[1], pz[2],
                    inverse_bin_width=inverse_bin_width)
        elif args.distribution_type == 'normal':
            if args.n_mixtures == 1:
                log_pz = log_discretized_normal(
                    z, *pz, inverse_bin_width=inverse_bin_width)
            else:
                log_pz = log_mixture_discretized_normal(
                    z, *pz, inverse_bin_width=inverse_bin_width)

    elif args.variable_type == 'continuous':
        if args.distribution_type == 'logistic':
            log_pz = log_logistic(z, *pz)
        elif args.distribution_type == 'normal':
            if args.n_mixtures == 1:
                log_pz = log_normal(z, *pz)
            else:
                log_pz = log_mixture_normal(z, *pz)

    log_pz = ops.ReduceSum()(
        log_pz,
        axis=[1, 2, 3])

    return log_pz


def convert_bpd(log_p, input_size):
    """convert log probability to bpd value"""
    return -log_p / (np.prod(input_size) * np.log(2.))


def compute_loss_array(pz, z, pys, ys, ldj, args):
    """compute loss and bpd given the arguments of flow model"""
    bpd_per_prior = []

    # Likelihood of final representation.
    log_pz = compute_log_pz(pz, z, args)
    bpd_per_prior.append(convert_bpd(log_pz, args.input_size))
    log_p = log_pz

    # Add likelihoods of intermediate representations.
    if ys:
        log_pys = compute_log_ps(pys, ys, args)
        for log_py in log_pys:
            log_p += log_py
            bpd_per_prior.append(convert_bpd(log_py, args.input_size))

    log_p += ldj

    loss = -log_p
    bpd = convert_bpd(log_p, args.input_size)

    return loss, bpd, bpd_per_prior
