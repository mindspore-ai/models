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
'''Train the gan model'''
import gc
import os.path
import time
import argparse
import numpy as np

def parameter_parser():
    '''parameter parser'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_start", default=-1, type=float)
    parser.add_argument("--sigma_end", default=0, type=float)
    parser.add_argument("-s", "--sigma", default=None)
    parser.add_argument("--batch_size_t", type=int, default=10, help="size of the test batches")
    parser.add_argument("--valid_path", type=str, default="../../dataprocess/preprocess_Result/valid_data.txt")
    parser.add_argument("--test_path", type=str, default="../../dataprocess/preprocess_Result/test_data.txt")
    parser.add_argument("--batch_size_v", type=int, default=1000, help="size of the valid batches")
    parser.add_argument("-c", "--cross_val", default=10, type=int, help="Number of cross valiation folds")
    parser.add_argument("--data_path", type=str, default="../../results/mxbase/")
    return parser.parse_args()

opt = parameter_parser()


def log_mean_exp(a):
    max_ = a.max(axis=1)
    max2 = max_.reshape((max_.shape[0], 1))
    return max_ + np.log(np.exp(a - max2).mean(1))


def mind_parzen(x, mu, sigma):
    a = (x.reshape((x.shape[0], 1, x.shape[-1])) - mu.reshape((1, mu.shape[0], mu.shape[-1]))) / sigma
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
    nlls = np.array([]).astype(np.float32)
    for i in range(n_batches):
        begin = time.time()
        nll = mind_parzen(x[inds[i::n_batches]], samples, sigma)
        end = time.time()
        times.append(end - begin)
        nlls = np.concatenate((nlls, nll))

        if i % 10 == 0:
            print(i, np.mean(times), nlls.mean())

    return nlls


def cross_validate_sigma(samples, data, sigmas, batch_size):
    '''cross_validate_sigma'''
    lls = np.array([]).astype(np.float32)
    print("cross_validate_sigma start")
    print("epoch is", len(sigmas))
    num = 0
    for sigma in sigmas:
        print("num is", num)
        num += 1
        print(sigma)
        tmp = get_nll(data, samples, sigma, batch_size=batch_size)
        tmp = tmp.mean()
        tmp = tmp.reshape((1, 1))
        tmp = np.squeeze(tmp, 1)
        lls = np.concatenate((lls, tmp))
        gc.collect()

    ind = lls.argmax()
    print("cross_validate_sigma over")
    return sigmas[ind]


def get_valid():
    '''get_valid'''
    image = np.loadtxt(opt.valid_path)
    print("get valid over")
    return image


def get_test():
    image = np.loadtxt(opt.test_path)
    print("get test over")
    return image


def parzen(samples):
    '''parzen'''
    ll = [1]
    se = 1
    print("parzan start")
    shape = samples.shape
    samples = samples.reshape((shape[0], -1))
    if opt.sigma is None:
        valid = get_valid()
        valid = valid / 255
        # valid = Tensor(valid)

        sigma_range = np.logspace(opt.sigma_start, opt.sigma_end, num=opt.cross_val)
        sigma = cross_validate_sigma(samples, valid, sigma_range, opt.batch_size_t)
    else:
        sigma = float(opt.sigma)

    print("Using Sigma: {}".format(sigma))
    gc.collect()

    test_data = get_test()
    test_data = test_data / 255
    ll = get_nll(test_data, samples, sigma, batch_size=opt.batch_size_t)
    se = ll.std() / np.sqrt(test_data.shape[0])

    print("Eval result:Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))

    return ll.mean(), se

def readdata(path):
    Data = []
    for i in range(10000):
        file = os.path.join(path, str(i) + ".bin")
        data = np.fromfile(file, dtype=np.float32).reshape(28, 28)
        Data.append(data)
    return np.array(Data)

if __name__ == '__main__':
    data_path = opt.data_path
    data1 = readdata(data_path)
    print(data1.shape)
    data1 = data1 * 127.5 + 127.5
    data1 = data1 / 255
    mean_ll, se_ll = parzen(data1)
