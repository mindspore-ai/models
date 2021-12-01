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
'''preprocess'''
import os
import numpy as np
from src.dataset import create_Dataset
from src.config import config as cfg

def gen_bin():
    '''generate binary files'''
    os.makedirs(os.path.join(cfg.pre_result_path, "image"), exist_ok=True)

    val_dataset, _ = create_Dataset(cfg.val_data_path, 0, cfg.export_batch_size,\
                                    1, 0, shuffle=False)
    data_loader = val_dataset.create_dict_iterator(output_numpy=True)
    labels_list = []
    label_path = cfg.label_path
    for i, data in enumerate(data_loader):
        file_name = "unet3p_bs" + str(cfg.export_batch_size) + "_" + str(i) + ".bin"
        file_path = os.path.join(cfg.pre_result_path, "image", file_name)
        data["image"].tofile(file_path)
        labels_list.append(data["mask"])
    np.save(label_path, labels_list)
    print("="*20, "export bin files finished", "="*20)

if __name__ == '__main__':
    gen_bin()
