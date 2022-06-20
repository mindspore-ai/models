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
"""  spade dataset init """

import importlib
import numpy as np
import mindspore.dataset as ds
from src.data.base_dataset import BaseDataset

def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "src.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

class DatasetInit:
    def __init__(self, opt):
        super(DatasetInit, self).__init__()
        self.opt = opt

    def create_dataset_distribute(self, instance, device_id, group_size):
        dataset = ds.GeneratorDataset(instance,
                                      ["label", "instance", "image"],
                                      shuffle=not self.opt.serial_batches,
                                      num_shards=group_size,
                                      shard_id=device_id,
                                      num_parallel_workers=8)
        dataset = dataset.batch(batch_size=self.opt.batchSize, drop_remainder=self.opt.isTrain, num_parallel_workers=8)
        dataset = dataset.map(operations=[self.preprocess_input],
                              input_columns=['label', 'instance'],
                              output_columns=['label', 'instance', 'input_semantics'],
                              column_order=["label", "instance", "image", 'input_semantics'],
                              num_parallel_workers=8)
        return dataset

    def create_dataset_not_distribute(self, instance):
        dataset = ds.GeneratorDataset(instance,
                                      ["label", "instance", "image"],
                                      shuffle=not self.opt.serial_batches,
                                      num_parallel_workers=8)
        dataset = dataset.batch(batch_size=self.opt.batchSize, drop_remainder=self.opt.isTrain, num_parallel_workers=8)
        dataset = dataset.map(operations=[self.preprocess_input],
                              input_columns=['label', 'instance'],
                              output_columns=['label', 'instance', 'input_semantics'],
                              column_order=["label", "instance", "image", 'input_semantics'],
                              num_parallel_workers=8)
        return dataset

    def get_edges_(self, t):
        edge = np.zeros(t.shape).astype(np.float32)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge

    def preprocess_input(self, label, instance):
        # create one-hot label map
        label_map = label
        bs, _, h, w = label_map.shape
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        eyes = np.eye(nc)
        input_semantics = eyes[label_map.reshape(bs, h, w).astype(np.int)].transpose((0, 3, 1, 2)).astype(np.float32)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = instance
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = np.concatenate((input_semantics, instance_edge_map), axis=1)

        return label, instance, input_semantics
