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
""" eval_callback """
import mindspore.dataset as ds
from src.dataset import pt_dataset
from src.dataset import pt_transform as transform
from src.model_utils.config import config as cfg

def create_dataset(purpose, data_root, data_list, device_num, rank_id):
    """ get dataset """
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    if purpose == 'train':
        cur_transform = transform.Compose([
            transform.RandScale([0.5, 2.0]),
            transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([473, 473], crop_type='rand', padding=mean, ignore_label=255),
            transform.Normalize(mean=mean, std=std, is_train=True)])
        data = pt_dataset.SemData(
            split=purpose, data_root=data_root,
            data_list=data_list,
            transform=cur_transform,
            data_name=cfg.data_name
        )
        dataset = ds.GeneratorDataset(data, column_names=["data", "label"],
                                      shuffle=True, num_shards=device_num, shard_id=rank_id)
        dataset = dataset.batch(cfg.batch_size, drop_remainder=False)
    else:
        cur_transform = transform.Compose([
            transform.Crop([473, 473], crop_type='center', padding=mean, ignore_label=255),
            transform.Normalize(mean=mean, std=std, is_train=True)])
        data = pt_dataset.SemData(
            split=purpose, data_root=data_root,
            data_list=data_list,
            transform=cur_transform,
            data_name=cfg.data_name
        )

        dataset = ds.GeneratorDataset(data, column_names=["data", "label"],
                                      shuffle=False, num_shards=device_num, shard_id=rank_id)
        dataset = dataset.batch(cfg.batch_size_val, drop_remainder=False)

    return dataset
