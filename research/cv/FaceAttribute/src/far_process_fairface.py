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
"""far process fairface."""

import os
import csv

if __name__ == '__main__':
    test_paths = ['train', 'val']
    for test_path in test_paths:
        root_path = './'
        jpg_path = os.path.join(root_path, test_path)
        mat_path = os.path.join(root_path, 'fairface_label_' + test_path + '.csv')
        target_txt = mat_path.replace('.csv', '.txt')
        if os.path.exists(target_txt):
            os.remove(target_txt)

        with open(target_txt, 'w') as txt:
            with open(mat_path, 'r') as f:
                readers = list(csv.reader(f))
                for img_name in os.listdir(jpg_path):
                    if not img_name.endswith('.jpg'):
                        continue
                    image_num = int(img_name.replace('.jpg', ''))
                    row = readers[image_num]
                    txt.write('./' + row[0] + ' ')

                    if row[1] == '0-2':
                        txt.write(str(0) + ' ')
                    elif row[1] == '3-9':
                        txt.write(str(1) + ' ')
                    elif row[1] == '10-19':
                        txt.write(str(2) + ' ')
                    elif row[1] == '20-29':
                        txt.write(str(3) + ' ')
                    elif row[1] == '30-39':
                        txt.write(str(4) + ' ')
                    elif row[1] == '40-49':
                        txt.write(str(5) + ' ')
                    elif row[1] == '50-59':
                        txt.write(str(6) + ' ')
                    elif row[1] == '60-69':
                        txt.write(str(7) + ' ')
                    elif row[1] == 'more than 70':
                        txt.write(str(8) + ' ')
                    else:
                        print(row[0], row[1])
                        txt.write(str(-1) + ' ')

                    if row[2] == 'Male':
                        txt.write(str(0) + ' ')
                    elif row[2] == 'Female':
                        txt.write(str(1) + ' ')
                    else:
                        print(row[0], row[2])
                        txt.write(str(-1) + ' ')

                    txt.write('1')
                    txt.write('\n')
