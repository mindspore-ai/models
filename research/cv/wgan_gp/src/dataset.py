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

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2

def create_dataset(dataroot, batchSize, imageSize, repeat_num=1, workers=8, target='Ascend'):

    # define map operations
    resize_op = C.Resize(imageSize)
    normalize_op = C.Normalize(mean=(0.5*255, 0.5*255, 0.5*255), std=(0.5*255, 0.5*255, 0.5*255))
    hwc2chw_op = C.HWC2CHW()

    data_set = ds.Cifar10Dataset(dataroot, num_parallel_workers=workers, shuffle=True)
    transform = [resize_op, normalize_op, hwc2chw_op]

    type_cast_op = C2.TypeCast(ms.int32)

    data_set = data_set.map(input_columns='image', operations=transform, num_parallel_workers=workers)
    data_set = data_set.map(input_columns='label', operations=type_cast_op, num_parallel_workers=workers)

    data_set = data_set.batch(batchSize, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set
