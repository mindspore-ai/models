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

import os
import numpy as np

from src.dataset import classification_dataset
from src.model_utils.config import config


def gen_bin(data_dir):
    '''generate numpy bin files'''
    config.load_type = 'test'
    dataset, _ = classification_dataset(config.batch_size, shuffle=False, repeat_num=1,
                                        drop_remainder=False)

    eval_datasize = dataset.get_dataset_size()
    print("test dataset length is:", eval_datasize)

    image_path = os.path.join(data_dir, "image")
    label_path = os.path.join(data_dir, "label_bs" + str(config.batch_size) + ".npy")
    label_list = []
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    for index, (input_data, label) in enumerate(dataset):
        f_name = "c3d_bs" + str(config.batch_size) + "_" + str(index) + ".bin"
        input_data.asnumpy().tofile(os.path.join(image_path, f_name))
        label_list.append(label.asnumpy())

    np.save(label_path, label_list)
    print('=' * 20, "export bin files finished", "=" * 20)


if __name__ == '__main__':
    gen_bin(data_dir=config.pre_result_path)
