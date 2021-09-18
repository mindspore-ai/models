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
from argparse import ArgumentParser
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from dataset import cityscapes_datapath

# example:
# python build_mrdata.py \
# --dataset_path /path/to/cityscapes/ \
# --subset train \
# --output_name train.mindrecord

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--subset', type=str)
    parser.add_argument('--output_name', type=str)
    config = parser.parse_args()

    output_name = config.output_name
    subset = config.subset
    dataset_path = config.dataset_path
    if not subset in ("train", "val"):
        raise RuntimeError('subset should be "train" or "val"')

    dataPathLoader = cityscapes_datapath(dataset_path, subset)

    writer = FileWriter(file_name=output_name)
    seg_schema = {"file_name": {"type": "string"}, "label": {"type": "bytes"}, "data": {"type": "bytes"}}
    writer.add_schema(seg_schema, "seg_schema")

    data_list = []
    cnt = 0
    for img_path, label_path in tqdm(dataPathLoader):
        sample_ = {"file_name": img_path.split('/')[-1]}

        with open(img_path, 'rb') as f:
            sample_['data'] = f.read()

        with open(label_path, 'rb') as f:
            sample_['label'] = f.read()
        data_list.append(sample_)
        cnt += 1
        if cnt % 100 == 0:
            writer.write_raw_data(data_list)
            print('number of samples written:', cnt)
            data_list = []

    if data_list:
        writer.write_raw_data(data_list)
    writer.commit()
    print('number of samples written:', cnt)
