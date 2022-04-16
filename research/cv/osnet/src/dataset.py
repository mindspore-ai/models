# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""OSNet dataset."""

import os
import mindspore.dataset as ds
from model_utils.transforms import build_train_transforms, build_test_transforms
from .datasets_define import Market1501, DukeMTMCreID, MSMT17, CUHK03


def init_dataset(name, **kwargs):
    """Initializes an image dataset."""
    __image_datasets = {
        'market1501': Market1501,
        'cuhk03': CUHK03,
        'dukemtmcreid': DukeMTMCreID,
        'msmt17': MSMT17,
    }
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __image_datasets[name](**kwargs)


def dataset_creator(
        root='',
        dataset=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        mode=None
):
    '''
    create and preprocess data for train and evaluate
    '''
    if dataset is None:
        raise ValueError('dataset must not be None')
    dataset_ = init_dataset(
        name=dataset,
        root=root,
        mode=mode,
        cuhk03_labeled=cuhk03_labeled,
        cuhk03_classic_split=cuhk03_classic_split,
    )

    num_pids = dataset_.num_train_pids

    if mode == 'train':
        sampler = ds.RandomSampler()
        device_num, rank_id = _get_rank_info()
        if device_num == 1:
            data_set = ds.GeneratorDataset(dataset_, ['img', 'pid'],
                                           sampler=sampler, num_parallel_workers=workers)
        else:
            data_set = ds.GeneratorDataset(dataset_, ['img', 'pid'], num_parallel_workers=workers,
                                           num_shards=device_num, shard_id=rank_id, shuffle=True)
        transforms = build_train_transforms(height=height, width=width, transforms=transforms,
                                            norm_mean=norm_mean, norm_std=norm_std)
        data_set = data_set.map(operations=transforms, input_columns=['img'])
        data_set = data_set.batch(batch_size=batch_size_train, drop_remainder=True)
        return num_pids, data_set


    data_set = ds.GeneratorDataset(dataset_, ['img', 'pid', 'camid'],
                                   num_parallel_workers=workers)
    transforms = build_test_transforms(height=height, width=width,
                                       norm_mean=norm_mean, norm_std=norm_std)
    data_set = data_set.map(operations=transforms, input_columns=['img'])
    data_set = data_set.batch(batch_size=batch_size_test, drop_remainder=False)

    return num_pids, data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
