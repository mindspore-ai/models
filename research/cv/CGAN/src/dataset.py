# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""create dataset"""

import os
import struct
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_rank, get_group_size


def create_dataset(data_path, usage='train', img_size=32, batch_size=32, repeat_size=1,
                   real_valued_mnist=False, num_parallel_workers=8):
    """
    create dataset for cnn
    Args:
        data_path(str): data directory
        usage(str): data type
        batch_size(int): batch_size
        repeat_size(int): repeat_size
        real_valued_mnist: mnist type
        num_parallel_workers(int): num_parallel_workers


    Returns:
      mnist_ds: mnist_dataset dataset, each dictionary has keys "image" and "label"
    """
    rank_size, rank_id = _get_rank_info()

    # define dataset
    if real_valued_mnist:
        mnist_ds = get_real_valued_mnist(data_path, num_parallel_workers=num_parallel_workers, usage=usage,
                                         num_shards=rank_size, shard_id=rank_id)
    else:
        mnist_ds = ds.MnistDataset(data_path, num_parallel_workers=num_parallel_workers, usage=usage,
                                   num_shards=rank_size, shard_id=rank_id)

    # define map operations
    resize_op = CV.Resize((img_size, img_size), interpolation=Inter.LINEAR)
    rescale_op = CV.Rescale(1.0 / 255.0, 0)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    mnist_ds = mnist_ds.shuffle(buffer_size=10000)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds

def load_mnist(images_path, labels_path):
    """
    Load real valued MNIST data
    Args:
        images_path(str): images data directory
        labels_path(str): labels data directory

    Returns:
        images: images
        labels: labels
    """
    with open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28, 1)
    return images, labels

def get_real_valued_mnist(dataset_dir, usage='all', out_as_numpy=False, num_samples=None, num_parallel_workers=1,
                          shuffle=None, sampler=None, num_shards=None, shard_id=None):
    """
    Load real valued MNIST dataset

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be "train", "test" or "all" . "train" will read from 60,000
        train samples, "test" will read from 10,000 test samples, "all" will read from all 70,000 samples.
        (default=None, will read all samples)
        out_as_numpy(bool): out_as_numpy
        num_samples (int, optional): The number of samples to be included in the dataset (default=None, all images).
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel (default=1).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            (default=None, expected order behavior shown in the table).
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            Random accessible input is required. When this argument is specified, 'num_samples' reflects the max
            sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None).

    Returns:
        dataset: a dataset with given data slices
    """
    if usage == 'train':
        images_path = os.path.join(dataset_dir, f'train-images-idx3-ubyte')
        labels_path = os.path.join(dataset_dir, f'train-labels-idx1-ubyte')
        images, labels = load_mnist(images_path, labels_path)
    elif usage == 'test':
        images_path = os.path.join(dataset_dir, f't10k-images-idx3-ubyte')
        labels_path = os.path.join(dataset_dir, f't10k-labels-idx1-ubyte')
        images, labels = load_mnist(images_path, labels_path)
    elif usage == 'all':
        images_path = os.path.join(dataset_dir, 'train', f'train-images-idx3-ubyte')
        labels_path = os.path.join(dataset_dir, 'train', f'train-labels-idx1-ubyte')
        images1, labels1 = load_mnist(images_path, labels_path)
        images_path = os.path.join(dataset_dir, 'test', f't10k-images-idx3-ubyte')
        labels_path = os.path.join(dataset_dir, 'test', f't10k-labels-idx1-ubyte')
        images2, labels2 = load_mnist(images_path, labels_path)
        images, labels = np.concatenate([images1, images2], axis=0), np.concatenate([labels1, labels2], axis=0)
    else:
        raise ValueError(f"Unknown usage: {usage}")

    data = (images, labels)
    if out_as_numpy:
        return data
    numpy_dataset = ds.NumpySlicesDataset(data, column_names=["image", "label"], num_samples=num_samples,
                                          num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                                          num_shards=num_shards, shard_id=shard_id)
    return numpy_dataset

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
