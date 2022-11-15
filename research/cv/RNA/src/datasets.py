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
"""Data operations, will be used in train.py and eval.py"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.common.dtype as mstype
import mindspore.ops as ops


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


class NormalizeByChannelMeanStd(nn.Cell):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, ms.Tensor):
            mean = ms.Tensor(mean)
        if not isinstance(std, ms.Tensor):
            std = ms.Tensor(std)
        self.mean = mean.view(1, 3, 1, 1)
        self.std = std.view(1, 3, 1, 1)
        self.sub = ops.Sub()
        self.div = ops.Div()
        self.print = ops.Print()

    def construct(self, tensor):
        tensor = self.sub(tensor, self.mean)
        tensor = self.div(tensor, self.std)
        return tensor

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def get_loaders(dir_, batch_size, dataset='cifar10', worker=4, norm=True):
    if norm:
        train_transform = [
            C.RandomCrop(32, padding=4),
            C.RandomHorizontalFlip(),
            C.Normalize(cifar10_mean, cifar10_std),
            C.HWC2CHW(),
        ]
        test_transform = [
            C.Normalize(cifar10_mean, cifar10_std),
            C.HWC2CHW(),
        ]
        dataset_normalization = None

    else:
        train_transform = [
            C.RandomCrop(32, padding=4),
            C.RandomHorizontalFlip(),
            C.HWC2CHW(),
        ]
        test_transform = [
            C.HWC2CHW(),
            C.Rescale(1/255., 0)
        ]
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=cifar10_mean, std=cifar10_std)

    if dataset == 'cifar10':
        train_dataset = ds.Cifar10Dataset(
            dir_, usage='train', num_parallel_workers=worker, shuffle=True)
        test_dataset = ds.Cifar10Dataset(
            dir_, usage='test', num_parallel_workers=worker, shuffle=False)
    elif dataset == 'cifar100':
        train_dataset = ds.Cifar100Dataset(
            dir_, usage='train', num_parallel_workers=worker, shuffle=True)
        test_dataset = ds.Cifar100Dataset(
            dir_, usage='test', num_parallel_workers=worker, shuffle=False)
    else:
        print('Wrong dataset:', dataset)
        exit()

    type_cast_op = C2.TypeCast(mstype.int32)

    if dataset == 'cifar10':
        train_dataset = train_dataset.map(operations=train_transform, input_columns="image",
                                          num_parallel_workers=worker)
        train_dataset = train_dataset.map(operations=type_cast_op, input_columns="label",
                                          num_parallel_workers=worker)
        test_dataset = test_dataset.map(operations=test_transform, input_columns="image",
                                        num_parallel_workers=worker)
        test_dataset = test_dataset.map(operations=type_cast_op, input_columns="label",
                                        num_parallel_workers=worker)
    elif dataset == 'cifar100':
        train_dataset = train_dataset.map(operations=train_transform, input_columns="image",
                                          num_parallel_workers=worker)
        train_dataset = train_dataset.map(operations=type_cast_op, input_columns="fine_label",
                                          num_parallel_workers=worker)
        test_dataset = test_dataset.map(operations=test_transform, input_columns="image",
                                        num_parallel_workers=worker)
        test_dataset = test_dataset.map(operations=type_cast_op, input_columns="fine_label",
                                        num_parallel_workers=worker)

    train_loader = train_dataset.batch(batch_size)
    test_loader = test_dataset.batch(batch_size)
    return train_loader, test_loader, dataset_normalization
