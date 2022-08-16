# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""dataset"""
import os
import mindspore.dataset as ds
import mindspore.common.dtype as mstype

import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import get_rank


from .data_manager import DatasetManager
from .data_loader import ImageDataset
from .transforms import Compose_Keypt, Resize_Keypt, ToTensor_Keypt, Normalize_Keypt, Random2DTranslation_Keypt, RandomHorizontalFlip_Keypt

def create_dataset(dataset_dir, root, width, height, keyptaware, heatmapaware, segmentaware, train_batch):
    """get dataset"""
    device_num, rank_id = _get_rank_info()

    data = DatasetManager(dataset_dir=dataset_dir, root=root)

    num_train_vids = data.num_train_vids
    num_train_vcolors = data.num_train_vcolors
    num_train_vtypes = data.num_train_vtypes

    trans = Compose_Keypt([
        Random2DTranslation_Keypt((width, height)),
        RandomHorizontalFlip_Keypt(),
        ToTensor_Keypt(),
        Normalize_Keypt(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(data.train, train_batch, (int)(train_batch / 8),
                           keyptaware=keyptaware,
                           heatmapaware=heatmapaware,
                           segmentaware=segmentaware,
                           transform=trans, imagesize=(width, height), is_train=True)

    if device_num == 1:
        dataloader = ds.GeneratorDataset(dataset,
                                         column_names=["img", "vid", "camid", "vcolor", "vtype", "vkeypt"],
                                         shuffle=True)
    else:
        dataloader = ds.GeneratorDataset(dataset,
                                         column_names=["img", "vid", "camid", "vcolor", "vtype", "vkeypt"],
                                         shuffle=True, num_shards=device_num, shard_id=rank_id)

    type_cast_float32_op = C2.TypeCast(mstype.float32)
    type_cast_int32_op = C2.TypeCast(mstype.int32)
    dataloader = dataloader.map(operations=type_cast_float32_op, input_columns="img")
    dataloader = dataloader.map(operations=type_cast_float32_op, input_columns="vkeypt")
    dataloader = dataloader.map(operations=type_cast_int32_op, input_columns="vid")
    dataloader = dataloader.map(operations=type_cast_int32_op, input_columns="camid")
    dataloader = dataloader.map(operations=type_cast_int32_op, input_columns="vcolor")
    dataloader = dataloader.map(operations=type_cast_int32_op, input_columns="vtype")
    dataloader = dataloader.batch(batch_size=train_batch, drop_remainder=True)

    return dataloader, num_train_vids, num_train_vcolors, num_train_vtypes

def eval_create_dataset(dataset_dir, root, width, height, keyptaware, heatmapaware, segmentaware, train_batch):
    """get eval dataset"""
    data = DatasetManager(dataset_dir=dataset_dir, root=root)

    num_train_vids = data.num_train_vids
    num_train_vcolors = data.num_train_vcolors
    num_train_vtypes = data.num_train_vtypes
    vcolor2label = data.vcolor2label
    vtype2label = data.vtype2label

    trans = Compose_Keypt([
        Resize_Keypt((width, height)),
        ToTensor_Keypt(),
        Normalize_Keypt(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    quary_dataset = ImageDataset(data.query, train_batch, (int)(train_batch / 8),
                                 keyptaware=keyptaware, heatmapaware=heatmapaware, segmentaware=segmentaware,
                                 transform=trans, imagesize=(width, height))
    gallery_dataset = ImageDataset(data.gallery, train_batch, (int)(train_batch / 8),
                                   keyptaware=keyptaware, heatmapaware=heatmapaware, segmentaware=segmentaware,
                                   transform=trans, imagesize=(width, height))

    quary_dataloader = ds.GeneratorDataset(quary_dataset,
                                           column_names=["img", "vid", "camid", "vcolor", "vtype", "vkeypt"],
                                           shuffle=False)
    gallery_dataloader = ds.GeneratorDataset(gallery_dataset,
                                             column_names=["img", "vid", "camid", "vcolor", "vtype", "vkeypt"],
                                             shuffle=False)

    type_cast_float32_op = C2.TypeCast(mstype.float32)
    type_cast_int32_op = C2.TypeCast(mstype.int32)

    quary_dataloader = quary_dataloader.map(operations=type_cast_float32_op, input_columns="img")
    quary_dataloader = quary_dataloader.map(operations=type_cast_float32_op, input_columns="vkeypt")
    quary_dataloader = quary_dataloader.map(operations=type_cast_int32_op, input_columns="vid")
    quary_dataloader = quary_dataloader.map(operations=type_cast_int32_op, input_columns="camid")
    quary_dataloader = quary_dataloader.map(operations=type_cast_int32_op, input_columns="vcolor")
    quary_dataloader = quary_dataloader.map(operations=type_cast_int32_op, input_columns="vtype")
    quary_dataloader = quary_dataloader.batch(batch_size=train_batch, drop_remainder=False, num_parallel_workers=8)

    gallery_dataloader = gallery_dataloader.map(operations=type_cast_float32_op, input_columns="img")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_float32_op, input_columns="vkeypt")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_int32_op, input_columns="vid")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_int32_op, input_columns="camid")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_int32_op, input_columns="vcolor")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_int32_op, input_columns="vtype")
    gallery_dataloader = gallery_dataloader.batch(batch_size=train_batch, drop_remainder=False, num_parallel_workers=8)

    return quary_dataloader, gallery_dataloader, num_train_vids,\
        num_train_vcolors, num_train_vtypes, vcolor2label, vtype2label

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = int(os.environ.get("DEVICE_ID"))

    return rank_size, rank_id
