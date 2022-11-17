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
"""preprocess"""
import os
import argparse

parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--data_path', type=str, default='', help='eval data dir')
args = parser.parse_args()

def create_label(result_path, dir_path):
    print("[WARNING] Create imagenet label. Currently only use for Imagenet2012!")
    text_path = os.path.join(result_path, "imagenet_label.txt")
    dirs = os.listdir(dir_path)
    file_list = []
    for file in dirs:
        file_list.append(file)
    file_list = sorted(file_list)
    total = 0
    img_label = {}
    text_file = open(text_path, 'a')
    for i, file_dir in enumerate(file_list):
        files = os.listdir(os.path.join(dir_path, file_dir))
        for f in files:
            img_label[f.split('.')[0]] = i
            line = f.split('.')[0] + ":" + str(i)
            text_file.write(line)
            text_file.write('\n')
        total += len(files)
    text_file.close()
    print("[INFO] Completed! Total {} data.".format(total))

if __name__ == "__main__":
    create_label('./preprocess_Result/', args.data_path)
        