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

'''
Convert imagenet id to label file
You can run this script by:
    python --dataset_path=/path/to/ImageNet2012/validation_preprocess
validation_preprocess/ should like this:

└─ validation_preprocess
  ├─ n01440764
    ├─ ILSVRC2012_val_00000293.JPEG
    ├─ ILSVRC2012_val_00002138.JPEG
    └─......
  ├─ n01440764
    └─......
  └─......

Then imgid2label.txt will be generated in current working directory
'''

import argparse
import os

parser = argparse.ArgumentParser(description="default name", add_help=False)

parser.add_argument("--dataset_path", type=str, default='',
                    help="imagenet absolute path like /path/to/ImageNet2012/validation_preprocess'")
parser.add_argument("--output_name", type=str, default='img_id_2_label.txt',
                    help="output file name, default is img_id_2_label.txt")

args, _ = parser.parse_known_args()
dataset_path = args.dataset_path
all_classes = os.listdir(dataset_path)
all_classes.sort()
label_list = []
for cls_idx, cls_name in enumerate(all_classes):
    cls_img_list = os.listdir(os.path.join(dataset_path, cls_name))
    cls_img_list.sort()
    for img in cls_img_list:
        img = img.split('.')[0]
        label_list.append(f'{img}:{cls_idx}\n')

with open(args.output_name, 'w') as fp:
    fp.writelines(label_list)
