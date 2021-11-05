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
Evaluate with CIFAR10 model.
"""
import os
import time
import argparse

import pickle
import numpy as np

from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor
import mindspore.numpy as mnp

from ivpf.model import Model
from ivpf.coder import Coder


context.set_context(mode=context.PYNATIVE_MODE)
context.set_context(device_target='GPU')

parser = argparse.ArgumentParser(description='Mindspore iVPF CIFAR10 coder')
parser.add_argument('--data_dir', required=True)
parser.add_argument('--n_bits', type=int, default=14)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--no_code', action='store_true', default=False)

coder_args = parser.parse_args()


class Args():
    """
    Arguments of the code.
    """

    def __init__(self):
        self.input_size = 32
        self.learn_split = False
        self.variable_type = 'discrete'
        self.distribution_type = 'normal'
        self.round_approx = 'smooth'
        self.coupling_type = 'shallow'
        self.conv_type = 'standard'
        self.densenet_depth = 12
        self.bottleneck = False
        self.n_channels = 512
        self.network1x1 = 'standard'
        self.auxilary_freq = -1
        self.actnorm = False
        self.LU = False
        self.coupling_lifting_L = True
        self.splitprior = True
        self.split_quarter = True
        self.splitfactor = 1
        self.n_levels = 3
        self.n_flows = 8
        self.n_mixtures = 5
        self.cond_L = True
        self.n_bits = 14


args = Args()
args.input_size = [3, 32, 32]
args.n_bits = coder_args.n_bits
if coder_args.no_code:
    args.variable_type = 'continuous'


def encode_batches(coder, images, batch_size):
    """encode batches"""
    fwd_r = []
    baselen = coder.ans_length()
    for idx in range(0, len(images), batch_size):
        x = Tensor.from_numpy(images[idx:idx + batch_size])

        tic = time.time()
        _, fm = coder.encode(x)
        toc = time.time()
        fwd_r.append(fm[1])

        codelen = coder.ans_length() - baselen
        bpd = codelen / ((idx + len(x)) * np.prod(args.input_size)
                         ) + 16. / np.prod(args.input_size)

        print(
            '[ENCODE] %5d, time: %.3fs, code bpd: %.4f' %
            (idx + len(x), toc - tic, bpd))

    return coder, fwd_r


def decode_batches(coder, fwd_r, images=None):
    """decode batches"""
    idx = len(images)
    imgs_rec = []
    for _ in range(len(fwd_r)):
        fm = fwd_r.pop()
        batch_size = len(fm)

        tic = time.time()
        _, x_rec = coder.decode(batch_size, [coder.n_coupling, fm])
        toc = time.time()
        x_rec = x_rec.asnumpy()
        imgs_rec.append(x_rec)

        idx -= batch_size
        if images is not None:
            x_ori = images[idx:idx + batch_size]
            error = np.sum(x_ori != x_rec)
            print('[DECODE] %5d, time: %.3fs, error %.2f' %
                  (idx + batch_size, toc - tic, error))
        else:
            print('[DECODE] %5d, time: %.3fs' % (idx + batch_size, toc - tic))

    return np.concatenate(imgs_rec)


def infer(model, images, batch_size):
    """perform inference of data batch given flow model"""
    total_bpd = 0.
    for idx in range(0, len(images), batch_size):
        x = Tensor.from_numpy(images[idx:idx + batch_size])

        tic = time.time()
        _, bpd, _, _, _, _, _, _ = model(x)
        toc = time.time()

        bpd = mnp.mean(bpd)
        total_bpd += bpd.asnumpy() * len(x)

        print('[INFERENCE] %5d, time: %.3fs, bpd %.4f, ave bpd: %.4f' %
              (idx + len(x), toc - tic, bpd.asnumpy(), total_bpd / (idx + len(x))))


def main():
    print('loading model ...')

    model = Model(args)
    model.set_train(False)
    model.set_grad(False)

    param_dict = load_checkpoint('model_weights/vpf_cifar10_f8_l3.ckpt')
    load_param_into_net(model, param_dict)

    coder = Coder(model)

    print('loading test data ...')

    file_path = os.path.join(
        coder_args.data_dir,
        'cifar-10-batches-py',
        'test_batch')
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    cifar10_test = entry['data'].reshape((-1, 3, 32, 32))

    if coder_args.no_code:
        print('perform inference on cifar10 data ...')
        infer(model, cifar10_test, coder_args.batch_size)
    else:
        print('encoding cifar10 data ...')
        coder, fwd_r = encode_batches(
            coder, cifar10_test, coder_args.batch_size)
        print('\ndecoding cifar10 data ...')
        decode_batches(coder, fwd_r, cifar10_test)
        print('\ncoding finished!')


if __name__ == '__main__':
    main()
