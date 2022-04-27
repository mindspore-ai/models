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
"""
preprocess.
"""

import os
import numpy as np

from eval import get_files
from src.dataset import create_dataset
from src.model_utils.config import config

def get_bin():
    '''generate bin files.'''

    test_path = os.path.join(config.data_path, 'output/test.lst')
    test_img = get_files(os.path.join(config.data_path, 'BSDS500/data/images/test'),
                         extension_filter='.jpg')
    test_label = get_files(os.path.join(config.data_path, 'BSDS500/data/labels/test'),
                           extension_filter='.jpg')
    f = open(test_path, "w")
    for img, label in zip(test_img, test_label):
        f.write(str(img) + " " + str(label))
        f.write('\n')

    # test.lst路径
    with open(test_path, 'r') as f:
        test_list = f.readlines()
    test_list = [i.rstrip() for i in test_list]


    test_dataset = create_dataset(test_path, is_training=False, is_shuffle=False)
    print("evaluation image number:", test_dataset.get_dataset_size())
    test_loader = test_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    print("Dataset creation Done!")

    bin_images_path = os.path.join(config.pre_result_path, "00_images")
    if not os.path.exists(bin_images_path):
        os.makedirs(bin_images_path)

    for idx, data in enumerate(test_loader):
        images = data['test'].astype(np.float32)
        filename, _ = test_list[idx].split()
        filename = filename.split('/')[-1]
        filename = filename.split('.')[0]
        print(filename)
        try:
            result_path_bin = os.path.join(bin_images_path, "{}.bin".format(filename))
            images.tofile(result_path_bin)
        except OSError:
            pass
    print("=" * 10, "export input bin files finished.", "=" * 10)


if __name__ == '__main__':
    get_bin()
