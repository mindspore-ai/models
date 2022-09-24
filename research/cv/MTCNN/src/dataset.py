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

from mindspore.dataset import MindDataset
try:
    from mindspore.dataset.vision import Decode, Normalize, HWC2CHW
except ImportError as error:
    from mindspore.dataset.vision.c_transforms import Decode, Normalize, HWC2CHW


def create_train_dataset(mindrecord_file, batch_size=128, device_num=1, rank_id=0, num_workers=8, do_shuffle=True):
    """Create MTCNN dataset with mindrecord for training"""
    ds = MindDataset(
        mindrecord_file,
        num_shards=device_num,
        columns_list=['image', 'label', 'box_target', 'landmark_target'],
        shard_id=rank_id,
        num_parallel_workers=num_workers,
        shuffle=do_shuffle
    )

    op_list = [Decode(), lambda x: x[:, :, (2, 1, 0)],
               Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]), HWC2CHW()]
    ds = ds.map(operations=op_list, input_columns=['image'])
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds
