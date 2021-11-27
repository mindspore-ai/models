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
"""mnist flash dataset"""
import os
import shutil
import numpy as np
from PIL import Image
from src.datasets.mnist_sampler import load_pkl, dump_pkl, load_mnist, \
    get_noisy_sampler, get_number_sampler, to_image, put_numbers

np.random.seed(1234)

def label_to_str(label):
    s = ''
    for i in range(10):
        if label & (1 << i):
            s += str(i)
        else:
            s += '_'
    return s


def get_numbers(label):
    numbers = []
    for i in range(10):
        if label & (1 << i):
            numbers.append(i)
    return numbers


def generate_dataset(data_dir, train=True, k=100, b=10, s=2):
    """generate mnist flash dataset"""
    img_sampler = get_noisy_sampler(data_dir)
    num_sampler = get_number_sampler(load_mnist(data_dir, 'train' if train else 'test'))

    data = []
    idx = 0
    for j in range(1 << b):
        for _ in range(k):
            if (idx + 1) % 1000 == 0 or idx + 1 == k * (1 << b):
                print('%d/%d' % (idx + 1, k * (1 << b)))
            idx += 1
            frames = []

            numbers = get_numbers(j)

            for i in numbers:
                for _ in range(np.random.randint(s) + 1):
                    img = img_sampler.sample((28, 28))
                    num = num_sampler.sample(i)
                    put_numbers(img, num)
                    frames.append(img)

            for i in range(25 - len(frames)):
                img = img_sampler.sample((28, 28))
                frames.append(img)

            np.random.shuffle(frames)
            video = np.array(frames)
            data.append((video, j))
    return data


def save_pkl(data, path):
    print('save pkl to: %s' % path)
    train_imgs, train_labels = list(zip(*data))
    dump_pkl((train_imgs, train_labels), path)


def save_images(data, image_dir):
    """save mnist images"""
    print('save videos to: %s' % image_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    idx = 0
    for video, label in data:
        img = np.zeros((5 * 28, 5 * 28), dtype='uint8')
        for i, frame in enumerate(video):
            x = i / 5
            y = i % 5
            img[x * 28:(x + 1) * 28, y * 28:(y + 1) * 28] = frame
        img = to_image(img)
        img.save(os.path.join(image_dir, '%s %06d.bmp' % (label_to_str(label), idx)))
        idx += 1


class MNISTFlash:
    """mnist flash dataset"""
    training_images_root = 'flash_train'
    test_images_root = 'flash_test'
    training_file = 'flash_train.pkl'
    test_file = 'flash_test.pkl'

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
            video, target = self.train_data[index], self.train_labels[index]
        else:
            video, target = self.test_data[index], self.test_labels[index]

        imgs = [Image.fromarray(img, mode='L') for img in video]

        if self.transform is not None:
            imgs = self.transform(imgs)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs, target

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def _check_exists(self):
        #         return True
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.test_file))

    def generate(self, force_generate):
        if self._check_exists() and not force_generate:
            return

        data = generate_dataset(self.root, train=True, k=100)
        save_pkl(data, os.path.join(self.root, self.training_file))

        data = generate_dataset(self.root, train=False, k=10)
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
