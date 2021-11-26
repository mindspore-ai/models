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
######################## create permuted mnist dataset ########################
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore import dtype as mstype

np.random.seed(0)
permuted = np.random.permutation(784)


def preprocess_MNIST(image, permute):
    '''for create permuted mnist'''
    data = image.copy()
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    output_data = data[:, permute]
    return output_data


def create_dataset(data_path, batch_size=64, repeat_size=1, num_parallel_workers=1, permute=permuted, is_train=True):
    """create permuted mnist dataset"""
    mnist_ds = ds.MnistDataset(data_path, shuffle=False)

    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)
    precess_mnist_op = (lambda image: preprocess_MNIST(image, permute))

    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    premuted_mnist_ds = mnist_ds.map(operations=precess_mnist_op, input_columns="image",
                                     num_parallel_workers=num_parallel_workers)

    # shuffle and batch
    buffer_size = 10000

    premuted_mnist_ds = premuted_mnist_ds.shuffle(buffer_size=buffer_size)
    premuted_mnist_ds = premuted_mnist_ds.batch(batch_size, drop_remainder=True)
    premuted_mnist_ds = premuted_mnist_ds.repeat(repeat_size)

    return premuted_mnist_ds
