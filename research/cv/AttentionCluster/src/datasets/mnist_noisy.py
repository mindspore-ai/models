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
"""mnist noisy dataset"""
import os
import shutil
import numpy as np
from PIL import Image
from src.datasets.mnist_sampler import load_pkl, dump_pkl, load_mnist, \
    get_noisy_sampler, get_number_sampler, to_image, put_numbers

np.random.seed(1234)

def generate_dataset(data_dir, train=True, n_noisy=30000, k=5):
    """generate mnist noisy dataset"""
    data = []
    img_sampler = get_noisy_sampler(data_dir)
    num_sampler = get_number_sampler(load_mnist(data_dir, 'train' if train else 'test'))
    for i in range(10):
        print('Generate for number %d ...' % i)
        idx = 0
        for _ in range(k):
            for num in num_sampler.numbers[i]:
                img = img_sampler.sample((28, 28))
                put_numbers(img, num)
                data.append((img, i))
                idx += 1
                if idx % 10000 == 0:
                    print(idx)
    print('Generate for noisy ...')
    for idx in range(n_noisy):
        img = img_sampler.sample((28, 28))
        data.append((img, 10))
        if (idx + 1) % 10000 == 0:
            print(idx + 1)
    return data


def save_pkl(data, path):
    print('save pkl to: %s' % path)
    train_imgs, train_labels = list(zip(*data))
    dump_pkl((train_imgs, train_labels), path)


def save_images(data, image_dir):
    print('save images to: %s' % image_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    idx = 0
    for img, label in data:
        img = to_image(img)
        img.save(os.path.join(image_dir, '%d_%06d.bmp' % (label, idx)))
        idx += 1


class MNISTNoisy:
    """mnist noisy dataset"""
    training_images_root = 'noisy_train'
    test_images_root = 'noisy_test'
    training_file = 'noisy_train.pkl'
    test_file = 'noisy_test.pkl'

    def __init__(self, root, train=True, transform=None, target_transform=None, generate=False,
                 force_generate=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if generate:
            self.generate(force_generate)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use generate=True to generate it')

        if self.train:
            self.train_data, self.train_labels = load_pkl(
                os.path.join(self.root, self.training_file))
        else:
            self.test_data, self.test_labels = load_pkl(
                os.path.join(self.root, self.test_file))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.test_file))

    def generate(self, force_generate):
        if self._check_exists() and not force_generate:
            return

        data = generate_dataset(self.root, train=True, n_noisy=30000, k=5)
        save_pkl(data, os.path.join(self.root, self.training_file))

        data = generate_dataset(self.root, train=False, n_noisy=1000, k=1)
        save_pkl(data, os.path.join(self.root, self.test_file))

    def save_image(self):
        """save image"""
        if self.train:
            image_root = os.path.join(self.root, self.training_images_root)
            if os.path.exists(image_root):
                shutil.rmtree(image_root)
            save_images(list(zip(self.train_data, self.train_labels)), image_root)
        else:
            image_root = os.path.join(self.root, self.test_images_root)
            if os.path.exists(image_root):
                shutil.rmtree(image_root)
            save_images(list(zip(self.test_data, self.test_labels)), image_root)
