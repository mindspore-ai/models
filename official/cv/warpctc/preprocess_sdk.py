
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

import os
import cv2
from src.model_utils.config import config as cf

data_path = "./data/test"
file_names = os.listdir(data_path)
file_list = [os.path.join(data_path, item) for item in file_names]

img_path = "./infer/Pre_Result"
if not os.path.isdir(img_path):
    os.makedirs(img_path)
label_dict = {}
for filename in file_list:
    index = filename.split("/")[-1].split(".")[0].split("-")[0]
    label_str = filename.split("/")[-1].split(".")[0].split("-")[-1]
    label = [int(i) for i in label_str]
    label.extend([int(10)] * (cf.max_captcha_digits - len(label)))
    img = cv2.imread(filename)
    new_name = str(index) + '.jpg'
    cv2.imwrite(os.path.join(img_path, new_name), img)
    label_dict[new_name] = label

with open('./infer/label.txt', 'w') as f:
    for k, v in label_dict.items():
        f.write(str(k) + ',' + str(v) + '\n')
