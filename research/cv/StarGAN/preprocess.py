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
"""pre process for 310 inference"""
import os

from src.utils import  create_labels
from src.config import get_config
from src.dataset import dataloader



if __name__ == "__main__":

    config = get_config()

    # Define Dataset

    data_path = config.celeba_image_dir
    attr_path = config.attr_path

    dataset, length = dataloader(img_path=data_path,
                                 attr_path=attr_path,
                                 batch_size=1,
                                 selected_attr=config.selected_attrs,
                                 dataset=config.dataset,
                                 mode='test',
                                 shuffle=False)

    img_path = os.path.join('../bin_data', "img_data")
    label_path = os.path.join('../bin_data', "label")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
        os.makedirs(label_path)
    ds = dataset.create_dict_iterator(num_epochs=1)
    print('Start preprocessing!')
    for idx, data in enumerate(ds):
        x_real = data['image']
        c_trg_list = create_labels(data['attr'].asnumpy(), selected_attrs=config.selected_attrs)
        for i in range(5):
            file_name = "sop_" + str(idx) + "_" + str(i) + ".bin"
            img_file_path = os.path.join(img_path, file_name)
            x_real.asnumpy().tofile(img_file_path)
            label_file_path = os.path.join(label_path, file_name)
            c_trg_list.asnumpy()[i].tofile(label_file_path)
            print('Finish processing img', idx, "saving as", file_name)
    print("=" * 20, "export bin files finished", "=" * 20)
