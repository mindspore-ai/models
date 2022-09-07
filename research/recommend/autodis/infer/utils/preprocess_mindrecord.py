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
sample script of preprocessing mindrecord data for autodis infer
"""
import os
import argparse
import numpy as np
import mindspore.dataset as ds

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file_path',
                    default='../data/input/mindrecord/test_input_part.mindrecord0',
                    help='input file path')
parser.add_argument('--dst_dir',
                    default='../data/input',
                    help='output folder')

args = parser.parse_args()

batch_size = 1000

data_set = ds.MindDataset(args.file_path, columns_list=['feat_ids', 'feat_vals', 'label'],
                          shuffle=False, num_parallel_workers=8)

data_set = data_set.map(operations=(lambda x, y, z: (np.array(x).flatten().reshape(batch_size, 39),
                                                     np.array(y).flatten().reshape(
                                                         batch_size, 39),
                                                     np.array(z).flatten().reshape(batch_size, 1))),
                        input_columns=['feat_ids', 'feat_vals', 'label'],
                        num_parallel_workers=8)
d = data_set.create_dict_iterator()

ids_arr = []
vals_arr = []
lab_arr = []
count = 0

for i, item in enumerate(d):
    ids_arr.extend(list(item['feat_ids'].asnumpy()[:]))
    vals_arr.extend(list(item['feat_vals'].asnumpy()[:]))
    lab_arr.extend(list(item['label'].asnumpy()[:]))
    count += batch_size

print("Have hadle {} lines".format(count))

ids_arr = np.array(ids_arr, dtype=np.int32)
vals_arr = np.array(vals_arr, dtype=np.float32)
lab_arr = np.array(lab_arr, dtype=np.float32)

ids_arr.tofile(os.path.join(args.dst_dir, 'ids.bin'))
vals_arr.tofile(os.path.join(args.dst_dir, 'wts.bin'))
lab_arr.tofile(os.path.join(args.dst_dir, 'label.bin'))
