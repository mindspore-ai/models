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
To convert the dataset from mindrecord format to binary format,
as required by the inference process.
"""
import os
import argparse
import numpy as np
import mindspore.dataset as ds

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file_path',
                    default='/home/admin/dataset/criteo/mindrecord/test_input_part.mindrecord0',
                    help='input file path')
parser.add_argument('--dst_dir',
                    default='./data/',
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

ids_arr = np.empty(shape=[4518000, 39], dtype=np.int32)
vals_arr = np.empty(shape=[4518000, 39], dtype=np.float32)
lab_arr = np.empty(shape=[4518000, 1], dtype=np.float32)

count = 0
start = 0
end = 1000
for item in d:
    count += 1
    ids_arr[start:end, :] = item['feat_ids'].asnumpy()[:]
    vals_arr[start:end, :] = item['feat_vals'].asnumpy()[:]
    lab_arr[start:end, :] = item['label'].asnumpy()[:]
    start = end
    end += 1000
    print('\n')
    print("count: ", count)

print(ids_arr.tofile(os.path.join(args.dst_dir, 'feat_ids.bin')))
print(vals_arr.tofile(os.path.join(args.dst_dir, 'feat_vals.bin')))
print(lab_arr.tofile(os.path.join(args.dst_dir, 'label.bin')))

np.savetxt('label.txt', lab_arr)
