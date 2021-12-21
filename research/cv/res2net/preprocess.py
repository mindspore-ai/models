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
"""preprocess."""
import os
import json
import numpy as np
from src.dataset import create_dataset1 as create_dataset
from src.model_utils.config import config

def cifar10_preprocess():
    '''convert cifar10 to numpy format'''
    target = config.device_target
    dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=config.batch_size,
                             eval_image_size=config.eval_image_size,
                             target=target)

    img_path = os.path.join(config.output_path, "img_data")
    label_path = os.path.join(config.output_path, "label.npy")
    os.makedirs(img_path, exist_ok=True)

    label_list = []
    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        img_data = data["image"]
        img_label = data["label"]

        file_name = "cifar10_bs" + str(config.batch_size) + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)

        label_list.append(img_label)
    np.save(label_path, label_list)

    print("[INFO] export bin files finished.")

def create_label():
    """create_label"""
    print("[WARNING] Create imagenet label. Currently only use for Imagenet2012!")
    file_path = config.data_path
    dirs = os.listdir(file_path)
    file_list = []
    for file in dirs:
        file_list.append(file)
    file_list = sorted(file_list)

    total = 0
    img_label = {}
    for i, file_dir in enumerate(file_list):
        files = os.listdir(os.path.join(file_path, file_dir))
        for f in files:
            img_label[f] = i
        total += len(files)
    json_path = os.path.join(config.output_path, "imagenet_label.json")
    with open(json_path, "w+") as label:
        json.dump(img_label, label)

    print("[INFO] Completed! Total {} data.".format(total))

if __name__ == '__main__':
    if config.dataset == "cifar10":
        cifar10_preprocess()
    else:
        create_label()
