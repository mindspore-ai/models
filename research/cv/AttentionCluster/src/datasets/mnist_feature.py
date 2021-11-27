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
"""mnist feature dataset"""
import os
import _pickle as cPickle


def load_pkl(path):
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f, encoding='bytes')
    finally:
        f.close()
    return rval


class MNISTFeature:
    """mnist feature dataset"""
    training_file = 'feature_train.pkl'
    test_file = 'feature_test.pkl'

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            self.train_data, self.train_labels, self.train_idxs = load_pkl(
                os.path.join(self.root, self.training_file))
        else:
            self.test_data, self.test_labels, self.test_idxs = load_pkl(
                os.path.join(self.root, self.test_file))

    def __getitem__(self, index):
        if self.train:
            feat, target, _ = self.train_data[index], self.train_labels[index], self.train_idxs[index]
        else:
            feat, target, _ = self.test_data[index], self.test_labels[index], self.test_idxs[index]

        if self.transform is not None:
            feat = self.transform(feat)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feat, target

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def _check_exists(self):
        print(os.path.join(self.root, self.training_file), os.path.join(self.root, self.test_file))
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.test_file))
