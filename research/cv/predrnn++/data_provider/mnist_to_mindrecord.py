# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter
from data_provider import mnist
from data_provider import preprocess

class MnistToRecord:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.dataset_name = input_param['dataset_name']
        self.file_name = input_param['name']
        self.shard_num = input_param['shard_num']
        self.patch_size = input_param['patch_size']
        self.seq_length = input_param['seq_length']
        self.input_length = input_param['input_length']
        self.img_channel = input_param['img_channel']
        self.img_width = input_param['img_width']
        self.batch_size = input_param['batch_size']
        self.mnist_handle = mnist

    def load_mnist_data(self):
        data_list = self.paths.split(',')
        input_param = {'paths': data_list,
                       'minibatch_size': 1,
                       'input_data_type': 'float32',
                       'is_output_sequence': True,
                       'name': self.dataset_name + ' my train iterator'}
        print(input_param)
        # load data
        self.data_handle = self.mnist_handle.InputHandle(input_param)
        self.data_handle.begin(do_shuffle=True)

    def convert_to_mindrecord(self):
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        if os.path.exists(self.file_name + ".db"):
            os.remove(self.file_name + ".db")

        writer = FileWriter(file_name=self.file_name, shard_num=self.shard_num)
        # define schema
        schema_json = {"input_x": {"type": "float32", "shape": [20, 16, 16, 16]}}
        writer.add_schema(schema_json, "It is a preprocesser for moving mnist dataset")

        data = []
        write_count = 0
        self.load_mnist_data()
        while not self.data_handle.no_batch_left():

            # pre-process
            ims = self.data_handle.get_batch()
            ims = preprocess.reshape_patch(ims, self.patch_size)
            sample = {}
            ims = np.squeeze(ims)
            sample['input_x'] = np.asarray(ims).astype(np.float32)

            data.append(sample)

            if len(data) % 100 == 0:
                writer.write_raw_data(data)
                write_count += 1
                data = []
            self.data_handle.next()

        if data:
            writer.write_raw_data(data)
        writer.commit()

    def read_mind_record(self, dataset_file, batch_size=8):
        data_set = ds.MindDataset(dataset_file=dataset_file, columns_list=["input_x"])
        for item in data_set.create_dict_iterator(output_numpy=True):
            np.save("firstinput_1.npy", item["input_x"])
        data_set = data_set.batch(batch_size, drop_remainder=True)
        return data_set

    def create_dataset(self, dataset_files="", rank_size=1, rank_id=0, do_shuffle="true", batch_size=8):
        """create train dataset"""

        data_set = ds.MindDataset(dataset_files, columns_list=["input_x"],
                                  num_shards=rank_size, shard_id=rank_id)
        # apply batch operations
        data_set = data_set.batch(batch_size, drop_remainder=True)
        return data_set

def create_mnist_dataset(dataset_files="", rank_size=1, rank_id=0, do_shuffle=True, batch_size=8):
    """create mnist dataset"""
    data_set = ds.MindDataset(dataset_files, columns_list=["input_x"], shuffle=do_shuffle)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set

def generate_mindrecord(file_path, mindrecord_name):
    input_param = {'paths': file_path,
                   'dataset_name': "mnist",
                   'name': mindrecord_name,
                   'batch_size': 1,
                   'seq_length': 20,
                   'input_length': 10,
                   'img_width': 64,
                   'img_channel': 1,
                   'shard_num': 1,
                   'patch_size': 4,
                   'sample_num': 10000,}
    record = MnistToRecord(input_param)
    record.convert_to_mindrecord()
