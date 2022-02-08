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

"""preprocess script"""

import os
from src.dataset import dataset_creator
from model_utils.config import config


def preprocess(result_path):
    '''preprocess data to .bin files for ascend310'''
    _, query_dataset = dataset_creator(root=config.data_path, height=config.height, width=config.width,
                                       dataset=config.target, norm_mean=config.norm_mean,
                                       norm_std=config.norm_std, batch_size_test=config.batch_size_test,
                                       workers=config.workers, cuhk03_labeled=config.cuhk03_labeled,
                                       cuhk03_classic_split=config.cuhk03_classic_split, mode='query')
    _, gallery_dataset = dataset_creator(root=config.data_path, height=config.height,
                                         width=config.width, dataset=config.target,
                                         norm_mean=config.norm_mean, norm_std=config.norm_std,
                                         batch_size_test=config.batch_size_test,
                                         workers=config.workers, cuhk03_labeled=config.cuhk03_labeled,
                                         cuhk03_classic_split=config.cuhk03_classic_split, mode='gallery')
    img_path = os.path.join(result_path, "img_data")
    label_path = os.path.join(result_path, "label")
    camid_path = os.path.join(result_path, "camlabel")
    os.makedirs(img_path)
    os.makedirs(label_path)
    os.makedirs(camid_path)

    for idx, data in enumerate(query_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["img"]
        img_label = data["pid"]
        img_cam_label = data["camid"]

        file_name = "query_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)

        label_file_path = os.path.join(label_path, file_name)
        img_label.tofile(label_file_path)

        camlabel_file_path = os.path.join(camid_path, file_name)
        img_cam_label.tofile(camlabel_file_path)

    for idx, data in enumerate(gallery_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["img"]
        img_label = data["pid"]
        img_cam_label = data["camid"]

        file_name = "gallery_" + str(config.target) + "_" + str(config.batch_size_test) + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)

        label_file_path = os.path.join(label_path, file_name)
        img_label.tofile(label_file_path)

        camlabel_file_path = os.path.join(camid_path, file_name)
        img_cam_label.tofile(camlabel_file_path)

    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == '__main__':
    preprocess(config.output_path)
