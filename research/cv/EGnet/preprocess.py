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
"""pre process for 310 inference"""
import os
from mindspore import DatasetHelper
from model_utils.config import base_config
from src.dataset import create_dataset

def preprocess(config):
    test_dataset, _ = create_dataset(config.test_batch_size, mode="test", num_thread=1,
                                     test_mode=config.test_mode, sal_mode=config.sal_mode,
                                     test_path=config.infer_path, test_fold=config.test_fold)
    dataset_helper = DatasetHelper(test_dataset, epoch_num=1, dataset_sink_mode=False)
    image_root = config.infer_image_root
    Names = []
    for data in os.listdir(image_root):
        name = data.split(".")[0]
        Names.append(name)
    Names = sorted(Names)
    for i, data_batch in enumerate(dataset_helper):
        sal_image, sal_label, _ = data_batch[0], data_batch[1], data_batch[2]
        file_name = Names[i]
        data_name = os.path.join("./preprocess_Result/", file_name + ".bin")
        mask_name = os.path.join("./preprocess_Mask_Result/", file_name + ".bin")
        sal_image.asnumpy().tofile(data_name)
        sal_label.asnumpy().tofile(mask_name)

if __name__ == "__main__":
    preprocess(base_config)
