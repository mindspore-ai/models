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
"""init dataset"""

import importlib
import mindspore.dataset as ds
from src.data.aligned_dataset import AlignedDataset
from src.data.base_dataset import BaseDataset
from src.data.single_dataset import SingleDataset

def find_dataset_using_name(dataset_name):
    """find_dataset_using_name"""
    # Given the option --dataset_mode [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "src.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
              % (dataset_filename, target_dataset_name))
        exit(0)

    return dataset


def get_option_setter(dataset_name):
    """get_option_setter"""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def dataset_instantiation(opt):
    """dataset_instantiation"""
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset(opt)
    print("dataset [%s] was created" % (instance.name()))
    return instance


def create_dataset(opt):
    """create_dataset"""
    dataset_instance = dataset_instantiation(opt)
    column_names = []
    if dataset_instance.name() == 'AlignedDataset':
        column_names.extend(['A', 'B', 'center', 'eyel_A', 'eyel_B', 'eyer_A', 'eyer_B', 'nose_A', 'nose_B', \
                            'mouth_A', 'mouth_B', 'hair_A', 'hair_B', 'bg_A', 'bg_B', 'mask', 'mask2', 'dt1gt', \
                            'dt2gt'])
    elif dataset_instance.name() == 'SingleImageDataset':
        column_names.extend(['A', 'A_path'])
    dataset = ds.GeneratorDataset(dataset_instance, column_names=column_names, num_shards=opt.group_size, \
                                  shard_id=opt.rank, shuffle=not opt.serial_batches, \
                                  num_parallel_workers=opt.num_parallel_workers)
    return dataset
