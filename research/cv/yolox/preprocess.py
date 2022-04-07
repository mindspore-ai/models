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
# =======================================================================================
"""
pre-process for inference
"""
import os
from tqdm import tqdm
from model_utils.config import config
from src.yolox_dataset import create_yolox_dataset


def preprocess():
    """
    generate img bin file

    """
    result_path = config.output_path
    data_root = os.path.join(config.data_dir, 'val2017')
    anno_file = os.path.join(config.data_dir, 'annotations/instances_val2017.json')
    dataset = create_yolox_dataset(data_root, anno_file, is_training=False, batch_size=1, device_num=1, rank=0)
    img_path = os.path.join(result_path, 'img_data')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    total_size = dataset.get_dataset_size()
    print("Total {} images to preprocess...".format(total_size))
    for _, data in enumerate(
            tqdm(dataset.create_dict_iterator(output_numpy=True, num_epochs=1), desc="Image preprocess",
                 total=total_size, unit="img", colour="GREEN")):
        image_data = data['image']
        img_info = data['image_shape'][0]
        img_id = data['img_id'][0]

        file_name = "{}_{}_{}.bin".format(str(img_id)[1:-1], str(img_info[0]), str(img_info[1]))
        img_file_path = os.path.join(img_path, file_name)
        image_data.tofile(img_file_path)

    print("img bin file generate finished, in %s" % img_path)


if __name__ == '__main__':
    preprocess()
