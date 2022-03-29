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
"""
Computes Kernel Inception Distance
Based on implementation from tensorflow:
https://github.com/taki0112/GAN_Metrics-Tensorflow
"""
import math
import os
from glob import glob

import mindspore as ms
import numpy as np
from PIL import Image

from src.metrics.inception import InceptionForDistance


def kernel_classifier_distance_and_std_from_activations(real_activations,
                                                        generated_activations,
                                                        max_block_size=10,
                                                        dtype=None):
    assert real_activations.ndim == 2
    assert generated_activations.ndim == 2

    if dtype is None:
        dtype = real_activations.dtype
        assert generated_activations.dtype == dtype
    else:
        real_activations = real_activations.astype(dtype)
        generated_activations = generated_activations.astype(dtype)

    # Figure out how to split the activations into blocks of approximately
    # equal size, with none larger than max_block_size.
    n_r = real_activations.shape[0]
    n_g = generated_activations.shape[0]

    n_bigger = max(n_r, n_g)
    n_blocks = int(math.ceil(n_bigger / max_block_size))

    v_r = n_r // n_blocks
    v_g = n_g // n_blocks

    n_plusone_r = n_r - v_r * n_blocks
    n_plusone_g = n_g - v_g * n_blocks

    sizes_r = np.concatenate((
        np.full((n_blocks - n_plusone_r), v_r),
        np.full((n_plusone_r), v_r + 1),
    ), 0)
    sizes_g = np.concatenate((
        np.full((n_blocks - n_plusone_g), v_g),
        np.full((n_plusone_g), v_g + 1),
    ), 0)

    zero = np.zeros((1,), dtype=np.int32)
    inds_r = np.concatenate((zero, np.cumsum(sizes_r)), 0)
    inds_g = np.concatenate((zero, np.cumsum(sizes_g)), 0)

    dim = real_activations.shape[1]

    def compute_kid_block(i):
        'Compute the ith block of the KID estimate.'
        r_s = inds_r[i]
        r_e = inds_r[i + 1]
        r = real_activations[r_s:r_e]
        m = (r_e - r_s).astype(dtype)

        g_s = inds_g[i]
        g_e = inds_g[i + 1]
        g = generated_activations[g_s:g_e]
        n = (g_e - g_s).astype(dtype)

        k_rr = (np.matmul(r, r.T) / dim + 1) ** 3
        k_rg = (np.matmul(r, g.T) / dim + 1) ** 3
        k_gg = (np.matmul(g, g.T) / dim + 1) ** 3
        return (-2 * np.mean(k_rg) +
                (np.sum(k_rr) - np.trace(k_rr)) / (m * (m - 1)) +
                (np.sum(k_gg) - np.trace(k_gg)) / (n * (n - 1)))

    ests = [compute_kid_block(x).astype(dtype) for x in range(n_blocks)]

    mn = sum(ests) / len(ests)

    # nn_impl.moments doesn't use the Bessel correction, which we want here
    if n_blocks <= 1:
        var = float("NaN")
    else:
        var = np.sum((ests - mn)**2 / (n_blocks - 1))

    return mn, math.sqrt(var / n_blocks)


def get_inception_activations(batch_size, images, inception_ckpt):
    n_batches = len(images) // batch_size

    net = InceptionForDistance()
    ms.load_checkpoint(inception_ckpt, net=net)

    act = np.zeros([n_batches * batch_size, 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = images[i] / 255. * 2 - 1
        input_tensor = ms.Tensor(np.expand_dims(inp, 0), dtype=ms.float32)
        res = net(input_tensor)
        act[i] = np.mean(res.asnumpy(), axis=(2, 3))
    return act


def get_images(filename):
    with open(filename, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((299, 299), Image.ANTIALIAS)
        return np.asarray(img)


def mean_kernel_inception_distance(output_path, dataset_dir, inception_ckpt):
    source_alpha = 0.98
    target_alpha = 1 - source_alpha

    filenames = glob(os.path.join(dataset_dir, 'testA', '*.*'))
    real_source_images = [get_images(filename) for filename in filenames]
    real_source_images = np.transpose(real_source_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join(dataset_dir, 'testB', '*.*'))
    real_target_images = [get_images(filename) for filename in filenames]
    real_target_images = np.transpose(real_target_images, axes=[0, 3, 1, 2])

    generated_dir = os.path.join(output_path, 'test')
    filenames = glob(os.path.join(generated_dir, 'A2B*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Three numpy arrays
    target_act = get_inception_activations(BATCH_SIZE, real_target_images, inception_ckpt)
    source_act = get_inception_activations(BATCH_SIZE, real_source_images, inception_ckpt)
    fake_act = get_inception_activations(BATCH_SIZE, fake_images, inception_ckpt)

    KID_mean, KID_stddev = kernel_classifier_distance_and_std_from_activations(target_act,
                                                                               fake_act,
                                                                               max_block_size=10)
    mean_KID_mean, mean_KID_stddev = kernel_classifier_distance_and_std_from_activations(source_act,
                                                                                         fake_act,
                                                                                         max_block_size=10)

    mean_KID_mean = (target_alpha * KID_mean + source_alpha * mean_KID_mean) / 2.0
    mean_KID_stddev = (target_alpha * KID_stddev + source_alpha * mean_KID_stddev) / 2.0

    print()

    print("mean_KID_mean : ", mean_KID_mean * 100)
    print("mean_KID_stddev : ", mean_KID_stddev * 100)
