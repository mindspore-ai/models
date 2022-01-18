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
"""far process RWMF."""

import os
from PIL import Image

if __name__ == '__main__':
    root_path = './Real-World-Masked-Face-Dataset-master'
    jpg_paths = ['RWMFD_part_1', 'RWMFD_part_2_pro']
    target_txt = 'RWMF_label_train.txt'
    if os.path.exists(target_txt):
        os.remove(target_txt)

    with open(target_txt, "w") as txt:
        for jpg_path in jpg_paths:
            cur_jpg_path = os.path.join(root_path, jpg_path)
            for img_dir in os.listdir(cur_jpg_path):
                cur_img_dir = os.path.join(cur_jpg_path, img_dir)
                for img_name in os.listdir(cur_img_dir):
                    if not img_name.endswith('.jpg'):
                        continue
                    img_path = os.path.join(cur_img_dir, img_name)
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except FileNotFoundError:
                        print('wrong img:', img_path)
                        continue
                    txt.write(img_path + ' ')
                    txt.write(str(-1) + ' ')
                    txt.write(str(-1) + ' ')
                    txt.write('0')
                    txt.write('\n')
