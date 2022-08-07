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
from src.mind_dataloader_final import get_test_loader
from src.model_utils.config import config

def preprocess():
    test_loader = get_test_loader(config.test_img_path, config.test_gt_path, batchsize=1, testsize=config.train_size)
    data_iterator = test_loader.create_tuple_iterator()
    total_test_step = 0
    test_data_size = test_loader.get_dataset_size()
    image_root = config.test_img_path
    Names = []
    for data in os.listdir(image_root):
        name = data.split(".")[0]
        Names.append(name)
    Names = sorted(Names)
    for imgs, targets in data_iterator:
        targets1 = targets.asnumpy()
        targets1 = targets1.astype(int)
        file_name = Names[total_test_step]
        data_name = os.path.join("./preprocess_Result/", file_name + ".bin")
        mask_name = os.path.join("./preprocess_Mask_Result/", file_name + ".bin")
        imgs.asnumpy().tofile(data_name)
        targets.asnumpy().tofile(mask_name)
        total_test_step = total_test_step + 1
        if total_test_step % 100 == 0:
            print("preprocess:{}/{}".format(total_test_step, test_data_size))

if __name__ == "__main__":
    preprocess()
