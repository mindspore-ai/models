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
"""preprocess"""
import ast
import json
import argparse
from pathlib import Path
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms as C2
from src.dataset.data_manager import DatasetManager
from src.dataset.data_loader import ImageDataset
from src.dataset.transforms import Compose_Keypt, Resize_Keypt, ToTensor_Keypt, Normalize_Keypt

parser = argparse.ArgumentParser(description='Eval MultiTaskNet')

parser.add_argument("--result_dir", type=str, help="")
parser.add_argument("--label_dir", type=str, help="")
parser.add_argument('--data_dir', type=str, default='')

parser.add_argument('--test-batch', default=1, type=int,
                    help="test batch size")
parser.add_argument('--heatmapaware', type=ast.literal_eval, default=False,
                    help="embed heatmaps to images")
parser.add_argument('--segmentaware', type=ast.literal_eval, default=False,
                    help="embed segments to images")

args = parser.parse_args()

if __name__ == '__main__':
    type_cast_float32_op = C2.TypeCast(mstype.float32)
    type_cast_int32_op = C2.TypeCast(mstype.int32)
    all_data = DatasetManager(dataset_dir='veri', root=args.data_dir)

    trans = Compose_Keypt([
        Resize_Keypt((256, 256)),
        ToTensor_Keypt(),
        Normalize_Keypt(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    query_dataset = ImageDataset(all_data.query, args.test_batch, 4,
                                 keyptaware=True,
                                 heatmapaware=args.heatmapaware,
                                 segmentaware=args.segmentaware,
                                 transform=trans,
                                 imagesize=(256, 256))
    gallery_dataset = ImageDataset(all_data.gallery, args.test_batch, 4,
                                   keyptaware=True,
                                   heatmapaware=args.heatmapaware,
                                   segmentaware=args.segmentaware,
                                   transform=trans,
                                   imagesize=(256, 256))

    query_dataloader = ds.GeneratorDataset(query_dataset,
                                           column_names=["img", "vid", "camid", "vcolor", "vtype", "vkeypt"],
                                           shuffle=False, num_shards=1, shard_id=0)
    gallery_dataloader = ds.GeneratorDataset(gallery_dataset,
                                             column_names=["img", "vid", "camid", "vcolor", "vtype", "vkeypt"],
                                             shuffle=False, num_shards=1, shard_id=0)

    query_dataloader = query_dataloader.map(operations=type_cast_float32_op, input_columns="img")
    query_dataloader = query_dataloader.map(operations=type_cast_float32_op, input_columns="vkeypt")
    query_dataloader = query_dataloader.map(operations=type_cast_int32_op, input_columns="vid")
    query_dataloader = query_dataloader.map(operations=type_cast_int32_op, input_columns="camid")
    query_dataloader = query_dataloader.map(operations=type_cast_int32_op, input_columns="vcolor")
    query_dataloader = query_dataloader.map(operations=type_cast_int32_op, input_columns="vtype")
    query_dataloader = query_dataloader.batch(batch_size=args.test_batch, drop_remainder=False)

    gallery_dataloader = gallery_dataloader.map(operations=type_cast_float32_op, input_columns="img")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_float32_op, input_columns="vkeypt")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_int32_op, input_columns="vid")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_int32_op, input_columns="camid")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_int32_op, input_columns="vcolor")
    gallery_dataloader = gallery_dataloader.map(operations=type_cast_int32_op, input_columns="vtype")
    gallery_dataloader = gallery_dataloader.batch(batch_size=args.test_batch, drop_remainder=False)

    query_path = args.result_dir + "query/"
    query_path_img = query_path + "img/"
    query_path_vkeypt = query_path + "vkeypt/"
    gallery_path = args.result_dir + "gallery/"
    gallery_path_img = gallery_path + "img/"
    gallery_path_vkeypt = gallery_path + "vkeypt/"

    label_query_list = {}
    for i, data in enumerate(query_dataloader.create_dict_iterator()):
        single_label_list = {}

        img = data["img"].asnumpy()
        vid = data["vid"].asnumpy()
        camid = data["camid"].asnumpy()
        vcolor = data["vcolor"].asnumpy()
        vtype = data["vtype"].asnumpy()
        vkeypt = data["vkeypt"].asnumpy()

        file_name_img = "veri_data_query_img" + "_" + str(i) + ".bin"
        file_path = query_path_img + file_name_img
        img.tofile(file_path)
        file_name_vkeypt = "veri_data_query_vkeypt" + "_" + str(i) + ".bin"
        file_path = query_path_vkeypt + file_name_vkeypt
        vkeypt.tofile(file_path)

        single_label_list['vid'] = vid.tolist()
        single_label_list['camid'] = camid.tolist()
        single_label_list['vcolor'] = vcolor.tolist()
        single_label_list['vtype'] = vtype.tolist()

        label_query_list['{}'.format(i)] = single_label_list

    label_gallery_list = {}
    for i, data in enumerate(gallery_dataloader.create_dict_iterator()):
        single_label_list = {}

        img = data["img"].asnumpy()
        vid = data["vid"].asnumpy()
        camid = data["camid"].asnumpy()
        vcolor = data["vcolor"].asnumpy()
        vtype = data["vtype"].asnumpy()
        vkeypt = data["vkeypt"].asnumpy()

        file_name_img = "veri_data_gallery_img" + "_" + str(i) + ".bin"
        file_path = gallery_path_img + file_name_img
        img.tofile(file_path)
        file_name_vkeypt = "veri_data_gallery_vkeypt" + "_" + str(i) + ".bin"
        file_path = gallery_path_vkeypt + file_name_vkeypt
        vkeypt.tofile(file_path)

        single_label_list['vid'] = vid.tolist()
        single_label_list['camid'] = camid.tolist()
        single_label_list['vcolor'] = vcolor.tolist()
        single_label_list['vtype'] = vtype.tolist()

        label_gallery_list['{}'.format(i)] = single_label_list

    label_list = {}
    label_list['num_train_vids'] = all_data.num_train_vids
    label_list['num_train_vcolors'] = all_data.num_train_vcolors
    label_list['num_train_vtypes'] = all_data.num_train_vtypes
    label_list['vcolor2label'] = all_data.vcolor2label
    label_list['vtype2label'] = all_data.vtype2label

    label_path = args.label_dir
    json_path = Path(label_path + 'label.json')
    with json_path.open('w') as json_path:
        json.dump(label_list, json_path)
    label_path = args.label_dir
    json_path = Path(label_path + 'query_label.json')
    with json_path.open('w') as json_path:
        json.dump(label_query_list, json_path)
    label_path = args.label_dir
    json_path = Path(label_path + 'gallery_label.json')
    with json_path.open('w') as json_path:
        json.dump(label_gallery_list, json_path)
