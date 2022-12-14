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
"""preprocess"""
import os
import numpy as np
import mindspore
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset as ds
from mindspore import dtype as mstype
from src.utils.opts import parse_opts
from src.datasets.dataset import get_val_set, get_test_set

if __name__ == '__main__':
    # init options
    opt = parse_opts()
    opt.video_path = opt.test_data_path
    print(opt)

    input_bin_path = os.path.join(opt.result_path, "00_data")
    label_bin_path = os.path.join(opt.result_path, "label")

    if not os.path.exists(input_bin_path):
        os.makedirs(input_bin_path)

    if not os.path.exists(label_bin_path):
        os.makedirs(label_bin_path)

    # create dataset
    if opt.dataset == 'kinetics':
        if opt.mode == 'multi':
            test_data = get_test_set(opt)
        else:
            test_data = get_val_set(opt)
        count = 0
        label = []
        for data in test_data:
            data[0].asnumpy().tofile(os.path.join(input_bin_path, str(count) + '.bin'))
            label.append(data[1].asnumpy())
            count += 1
        label = np.stack(label)
        label.tofile(os.path.join(label_bin_path, 'label.bin'))
    else:
        type_cast_op = mindspore.dataset.transforms.c_transforms.TypeCast(mstype.int32)
        transform_test = [
            C.Rescale(1.0 / 255.0, 0.0),
            C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            C.HWC2CHW()
        ]
        test_data = ds.Cifar10Dataset(dataset_dir=opt.test_data_path, usage='test', shuffle=False,
                                      num_parallel_workers=opt.n_threads)
        test_data = test_data.map(transform_test, input_columns=["image"])
        test_data = test_data.map(type_cast_op, input_columns=["label"])
        test_data = test_data.batch(opt.batch_size, drop_remainder=True)
        input_data = []
        label = []
        for data in test_data:
            input_data.append(data[0].asnumpy())
            label.append(data[1].asnumpy())
        input_data = np.stack(input_data).reshape(-1, 3, 32, 32)
        label = np.stack(label).reshape(-1, 1)
        input_data.tofile(os.path.join(input_bin_path, "input.bin"))
        label.tofile(os.path.join(label_bin_path, "label.bin"))
