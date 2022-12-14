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
"""pre_process"""
import os
import numpy as np
from src.dataset import create_dataset
from src.config import config as cfg


if __name__ == '__main__':
    eval_dataset = create_dataset(data_dir=cfg.dataset_path, config=cfg, shuffle=False, do_trains='val',
                                  num_worker=cfg.device_num,
                                  )
    data_path = os.path.join(cfg.result_path, 'data')
    os.makedirs(data_path)
    label_list = []
    for i, data in enumerate(eval_dataset.create_dict_iterator(output_numpy=True)):
        file_name = "stnet_data_bs" + str(cfg.batch_size) + "_" + str(i) + ".bin"
        file_path = data_path + '/' + file_name
        data['data'].tofile(file_path)
        label_list.append(data['label'])

    np.save(cfg.result_path + "label_ids.npy", label_list)
    print("tansfer_data down")
