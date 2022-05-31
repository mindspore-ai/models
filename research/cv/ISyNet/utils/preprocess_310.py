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
"""preprocess IsyNet 310."""
import os
import argparse

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2

PARSER = argparse.ArgumentParser(description="ISyNet preprocess")
PARSER.add_argument("--data_path", type=str, required=True, help="data path.")
PARSER.add_argument("--output_path", type=str, required=True, help="output path.")
ARGS = PARSER.parse_args()

def create_dataset(dataset_path, repeat_num=1, batch_size=1):
    """
    create a train or eval imagenet2012 dataset for IsyNet

    Args:
        dataset_path(string): the path of dataset.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    ds.config.set_prefetch_size(64)

    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    trans = [
        C.Decode(),
        C.Resize(256),
        C.CenterCrop(224),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set

if __name__ == '__main__':
    # create dataset
    DATASET = create_dataset(dataset_path=ARGS.data_path, batch_size=1)
    STEP_SIZE = DATASET.get_dataset_size()

    IMG_PATH = os.path.join(ARGS.output_path, "img_data")
    LABEL_PATH = os.path.join(ARGS.output_path, "label")
    os.makedirs(IMG_PATH)
    os.makedirs(LABEL_PATH)

    for idx, data in enumerate(DATASET.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["image"]
        img_label = data["label"]

        file_path = "imagenet_" + str(idx)
        img_file_path = os.path.join(IMG_PATH, file_path)
        os.makedirs(img_file_path)
        file_name = os.path.join(img_file_path, "input_0.bin")
        img_data.tofile(file_name)

        label_file_path = os.path.join(LABEL_PATH, "imagenet_" + str(idx) + "_label.bin")
        img_label.tofile(label_file_path)

    print("=" * 20, "export bin files finished", "=" * 20)
