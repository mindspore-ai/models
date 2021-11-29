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
import argparse
import numpy as np

from src.config import get_config
from src.datasets import get_train_test_dataset

parser = argparse.ArgumentParser(description="preprocess")
parser.add_argument("--train_data_dir", type=str, default="",
                    help="training dataset directory")
parser.add_argument("--test_data_dir", type=str, default="",
                    help="testing dataset directory")
parser.add_argument("--pre_result_dir", type=str, default="./preprocess_Result",
                    help="preprocess data path")
args_opt = parser.parse_args()

if __name__ == '__main__':
    config = get_config()

    train_data_dir = args_opt.train_data_dir
    test_data_dir = args_opt.test_data_dir

    print("start create dataset...")

    # eval_dataset contains train dataset and test dataset, which is used for knn eval
    eval_dataset = get_train_test_dataset(train_data_dir=train_data_dir, test_data_dir=test_data_dir,
                                          batchsize=config.batch_size, epoch=1)

    image_path = os.path.join(args_opt.pre_result_dir, "00_data")
    label_path = os.path.join(args_opt.pre_result_dir, "label.npy")
    training_path = os.path.join(args_opt.pre_result_dir, "training.npy")
    os.makedirs(image_path, exist_ok=True)
    label_list = []
    training_list = []
    for i, data in enumerate(eval_dataset.create_dict_iterator(output_numpy=True)):
        file_name = "ava_bs" + str(config.batch_size) + "_" + str(i) + ".bin"
        file_path = os.path.join(image_path, file_name)
        data["image"].tofile(file_path)
        label_list.append(data["label"])
        training_list.append(data["training"])
    np.save(os.path.join(label_path), label_list)
    np.save(os.path.join(training_path), training_list)
    print("=" * 20, "export bin files finished", "=" * 20)
