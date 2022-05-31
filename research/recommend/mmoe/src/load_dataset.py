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
"""data loader"""
import os
import mindspore.dataset as de
from mindspore.communication.management import get_rank, get_group_size
import mindspore.dataset.transforms as C
import mindspore.common.dtype as mstype


def _get_rank_info(run_distribute):
    """get rank size and rank id"""
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if run_distribute:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0
    return rank_size, rank_id


def create_dataset(data_path,
                   batch_size=32,
                   training=True,
                   target="Ascend",
                   run_distribute=False):
    """create dataset for train or eval"""
    if target == "Ascend":
        device_num, rank_id = _get_rank_info(run_distribute)

    if training:
        input_file = data_path + "train.mindrecord"
    else:
        input_file = data_path + "eval.mindrecord"

    if target != "Ascend" or device_num == 1:
        if training:
            ds = de.MindDataset(input_file,
                                columns_list=[
                                    'data', 'income_labels', 'married_labels'],
                                num_parallel_workers=8,
                                shuffle=True)
        else:
            ds = de.MindDataset(input_file,
                                columns_list=[
                                    'data', 'income_labels', 'married_labels'],
                                num_parallel_workers=8,
                                shuffle=False)
    else:
        if training:
            ds = de.MindDataset(input_file,
                                columns_list=[
                                    'data', 'income_labels', 'married_labels'],
                                num_parallel_workers=4,
                                shuffle=True,
                                num_shards=device_num,
                                shard_id=rank_id)
        else:
            ds = de.MindDataset(input_file,
                                columns_list=[
                                    'data', 'income_labels', 'married_labels'],
                                num_parallel_workers=4,
                                shuffle=False,
                                num_shards=device_num,
                                shard_id=rank_id)
    if target == 'Ascend':
        ds_label = [
            C.TypeCast(mstype.float16)
        ]
        ds = ds.map(operations=ds_label, input_columns=["data"])
        ds = ds.map(operations=ds_label, input_columns=["income_labels"])
        ds = ds.map(operations=ds_label, input_columns=["married_labels"])
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


if __name__ == '__main__':
    create_dataset(data_path='../data/')
