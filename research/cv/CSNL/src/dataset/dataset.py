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
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size
from src.dataset.div2k import DIV2K

def create_dataset_DIV2K(args, train=True, benchmark=False):
    train_dataset = DIV2K(args, train=train, benchmark=benchmark)
    t = train_dataset[0]
    print(type(t))
    print("total dataset length:", len(train_dataset))
    if args.distribute:
        rank_id = get_rank()
        group_size = get_group_size()
        data_set = ds.GeneratorDataset(train_dataset, ["LR", "HR"],
                                       shuffle=True, num_parallel_workers=1, num_shards=group_size, shard_id=rank_id)
    else:
        data_set = ds.GeneratorDataset(train_dataset, ["LR", "HR"], shuffle=True, num_parallel_workers=4)
    print("distributed dataset length:", data_set.get_dataset_size())
    data_set = data_set.batch(args.batch_size, drop_remainder=True)
    data_loader = data_set.create_dict_iterator(output_numpy=True)
    return data_loader
