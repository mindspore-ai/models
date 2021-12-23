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
'''Train the gan model'''
import gc
import os
import time
import numpy as np
from src.dataset import create_dataset_valid, load_test_data
from src.gan import Generator
from src.param_parse import parameter_parser

from mindspore import context
import mindspore.numpy
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net

set_seed(1)

os.makedirs("images", exist_ok=True)

opt = parameter_parser()
print(opt)


reshape = mindspore.ops.Reshape()
log = mindspore.ops.Log()
exp = mindspore.ops.Exp()
cat = mindspore.ops.Concat()
cat2 = mindspore.ops.Concat(1)
squeeze1 = mindspore.ops.Squeeze(1)


def log_mean_exp(a):
    max_ = a.max(axis=1)
    max2 = reshape(max_, (max_.shape[0], 1))
    return max_ + log(exp(a - max2).mean(1))


def mind_parzen(x, mu, sigma):
    a = (reshape(x, (x.shape[0], 1, x.shape[-1])) - reshape(mu, (1, mu.shape[0], mu.shape[-1]))) / sigma
    a5 = -0.5 * (a ** 2).sum(2)
    E = log_mean_exp(a5)
    t4 = sigma * np.sqrt(np.pi * 2)
    t5 = np.log(t4)
    Z = mu.shape[1] * t5
    return E - Z


def get_nll(x, samples, sigma, batch_size=10):
    '''get_nll'''
    inds = range(x.shape[0])
    inds = list(inds)
    n_batches = int(np.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = Tensor(np.array([]).astype(np.float32))
    for i in range(n_batches):
        begin = time.time()
        nll = mind_parzen(x[inds[i::n_batches]], samples, sigma)
        end = time.time()
        times.append(end - begin)
        nlls = cat((nlls, nll))

        if i % 10 == 0:
            print(i, np.mean(times), nlls.mean())

    return nlls


def cross_validate_sigma(samples, data, sigmas, batch_size):
    '''cross_validate_sigma'''
    lls = Tensor(np.array([]).astype(np.float32))
    for sigma in sigmas:
        print(sigma)
        tmp = get_nll(data, samples, sigma, batch_size=batch_size)
        tmp = tmp.mean()
        tmp = reshape(tmp, (1, 1))
        tmp = squeeze1(tmp)
        lls = cat((lls, tmp))
        gc.collect()

    ind = lls.argmax()
    return sigmas[ind]


def get_valid(limit_size=-1, fold=0):
    '''get_valid'''
    os.makedirs("./data/mnist", exist_ok=True)
    dataset = create_dataset_valid(batch_size=opt.batch_size_v, repeat_size=1, latent_size=opt.latent_dim)
    data = []
    for image in enumerate(dataset):
        data = image
        break
    image = data[1]
    image = image[0]
    image = reshape(image, (image.shape[0], 784))
    return image


def get_test(limit_size=-1, fold=0):
    dataset = load_test_data().astype("float32")
    image = Tensor(dataset)
    image = reshape(image, (image.shape[0], 784))
    return image


def parzen(samples):
    '''parzen'''
    ll = [1]
    se = 1
    shape = samples.shape
    samples = reshape(samples, (shape[0], -1))
    if opt.sigma is None:
        valid = get_valid(limit_size=1000, fold=0)
        valid = valid.asnumpy()
        valid = valid / 255
        valid = Tensor(valid)

        sigma_range = np.logspace(opt.sigma_start, opt.sigma_end, num=opt.cross_val)
        sigma = cross_validate_sigma(samples, valid, sigma_range, opt.batch_size_t)
    else:
        sigma = float(opt.sigma)

    print("Using Sigma: {}".format(sigma))
    gc.collect()

    test_data = get_test(limit_size=1000, fold=0)
    test2 = test_data.asnumpy()
    test_data = test_data / 255
    test2 = test2 / 255
    ll = get_nll(test_data, samples, sigma, batch_size=opt.batch_size_t)
    se = ll.std() / np.sqrt(test_data.shape[0])

    print("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))

    return ll.mean(), se


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

test_latent_code_parzen = Tensor(np.random.normal(size=(10000, opt.latent_dim)), dtype=mstype.float32)

if __name__ == '__main__':
    generator = Generator(opt.latent_dim)
    param_dict = load_checkpoint(opt.ckpt_path)
    load_param_into_net(generator, param_dict)
    imag = generator(test_latent_code_parzen)
    imag = imag * 127.5 + 127.5
    sigmoid = mindspore.ops.Sigmoid()

    samples1 = generator(test_latent_code_parzen)

    samples1 = samples1.asnumpy()
    samples1 = samples1 * 127.5 + 127.5
    samples1 = samples1 / 255
    samples2 = Tensor(samples1)

    mean_ll, se_ll = parzen(samples2)
    print("Log-Likelihood of test set = {}, se: {}".format(mean_ll, se_ll))
