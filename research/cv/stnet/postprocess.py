# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""post_process"""
import os
import numpy as np
from src.config import config as cfg


if __name__ == '__main__':
    rst_path = cfg.result_dir
    batch_size = 1
    labels = np.load(cfg.label_dir, allow_pickle=True)
    success_num = 0.0
    total_num = 0.0
    acc = 0.0
    for idx, label in enumerate(labels):
        f_name = os.path.join(rst_path, "stnet_data_bs" + str(batch_size) + "_" + str(idx) + "_0.bin")
        pred = np.fromfile(f_name, np.float32)
        pred = pred.reshape(batch_size, int(pred.shape[0] / batch_size))
        pred = np.argmax(pred, axis=1)[0]
        total_num = total_num + 1
        if pred == labels[idx]:
            success_num = success_num + 1
    acc = success_num / total_num
    print("success_num: ", success_num)
    print("total_num: ", total_num)
    print("acc: ", acc)
