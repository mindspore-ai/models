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

"""Download dataset"""

import os
import zipfile

import requests

from model_utils.config import config


def download(save_path, url):
    r = requests.get(url, allow_redirects=True)
    with open(save_path, "wb") as f:
        f.write(r.content)


def download_coco(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train2017_file_path = os.path.join(save_path, "train2017.zip")
    val2017_file_path = os.path.join(save_path, "val2017.zip")
    annotations_trainval2017_file_path = os.path.join(save_path, "annotations_trainval2017.zip")

    download(train2017_file_path, "http://images.cocodataset.org/zips/train2017.zip")
    download(val2017_file_path, "http://images.cocodataset.org/zips/val2017.zip")
    download(annotations_trainval2017_file_path,
             "http://images.cocodataset.org/annotations/annotations_trainval2017.zip")

    zipfile.ZipFile(train2017_file_path).extractall(save_path)
    zipfile.ZipFile(val2017_file_path).extractall(save_path)
    zipfile.ZipFile(annotations_trainval2017_file_path).extractall(save_path)


if __name__ == '__main__':
    coco_root = os.path.join(config.data_path, config.coco_root)
    if config.dataset == 'coco':
        download_coco(coco_root)
