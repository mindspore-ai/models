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
"""mnist sampler"""
import os
import _pickle as cPickle
import numpy as np
from PIL import Image
from src.datasets.mnist import create_dataset


def load_pkl(path):
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval


def dump_pkl(obj, path):
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, -1)
    finally:
        f.close()


def load_mnist(data_dir, usage='train'):
    train_datas = create_dataset(data_dir, usage)
    return [(data[0].squeeze().asnumpy(), data[1].asnumpy()[0]) for data in train_datas]


class NoisySampler:
    """Noisy Sampler"""
    def __init__(self, cnt=None):
        self.cnt = cnt if cnt is not None else {}

    def add(self, img):
        for x in np.ndarray.flatten(img):
            self.cnt[x] = self.cnt.get(x, 0) + 1

    def sample(self, shape):
        a = list(self.cnt.keys())
        tot = sum(self.cnt.values()) * 1.0
        p = [self.cnt[x] / tot for x in a]
        img = np.random.choice(a, size=shape, p=p)
        return img


def get_noisy_sampler(data_dir):
    """generate Noisy Sampler"""
    mnist_byte_sampler_path = os.path.join(data_dir, 'byte_sampler.pkl')
    sampler = None
    if os.path.exists(mnist_byte_sampler_path):
        sampler = load_pkl(mnist_byte_sampler_path)
    else:
        sampler = NoisySampler()
        data = load_mnist(data_dir, usage='all')
        for idx, (img, _) in enumerate(data):
            if idx % 1000 == 0:
                print(idx)
            sampler.add(img)
        dump_pkl(sampler, mnist_byte_sampler_path)
    return sampler


class NumberSampler:
    def __init__(self):
        self.numbers = {i: [] for i in range(10)}

    def add(self, img, label):
        self.numbers[label].append(img)

    def sample(self, label):
        return self.numbers[label][np.random.randint(len(self.numbers[label]))]


def get_number_sampler(data):
    sampler = NumberSampler()
    for img, label in data:
        sampler.add(img, label)
    return sampler


def to_image(img):
    return Image.fromarray(img)


def put_numbers(img, num, x=0, y=0):
    assert x >= 0 and x + num.shape[0] <= img.shape[0], 'invalid x: %d' % x
    assert y >= 0 and y + num.shape[1] <= img.shape[1], 'invalid y: %d' % y
    img[x:x + num.shape[0], y:y + num.shape[1]] = np.maximum(img[x:x + num.shape[0], y:y + num.shape[1]], num)
