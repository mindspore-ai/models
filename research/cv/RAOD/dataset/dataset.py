# Copyright 2023 Huawei Technologies Co., Ltd
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
# =======================================================================================
import os
import glob
import gzip
import numpy as np

import mindspore.dataset as ds


class DataGenerator:
    def __init__(self, args):
        path = args.in_path
        if os.path.isdir(path):
            print('Directory input path')
            self.flist = glob.glob(os.path.join(path, '*.raw'))
        else:
            print('txt input path')
            assert path.endswith('.txt'), f'Invalid input path < {path} >!'
            parent = os.path.dirname(path)
            with open(path) as fid:
                self.flist = [os.path.join(parent, 'raw', f.strip()) for f in fid.readlines()]
        self.length = len(self.flist)
        print(f'\n\n{self.length} raw images found')

        self.input_shape = tuple(args.input_shape)
        print(f'input shape: {self.input_shape}')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        idx = index % self.length

        fn = self.flist[idx]
        raw = self.get_raw(fn)

        return raw, os.path.basename(fn)

    @staticmethod
    def get_npy(fn):
        with gzip.GzipFile(fn.replace('raw', 'npy.gz')) as f:
            raw = np.load(f)
        return raw.transpose(2, 0, 1).astype(np.float32)

    def get_raw(self, fn):
        raw = np.fromfile(fn, dtype=np.uint8)
        raw = raw[0::3] + raw[1::3] * 256 + raw[2::3] * 65536
        raw = raw.reshape((1,) + self.input_shape).astype(np.float32)
        return raw


def create_dataset(args, is_train=False):
    data = ds.GeneratorDataset(DataGenerator(args),
                               python_multiprocessing=False,
                               column_names=['raw', 'file_name'],
                               num_parallel_workers=16, shuffle=is_train)
    data = data.batch(args.batch_size, drop_remainder=not is_train)
    return data
