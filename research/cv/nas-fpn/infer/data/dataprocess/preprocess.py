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

import argparse
import glob
import os

parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--data_path', type=str, default='', help='eval data dir')
parser.add_argument('--out_path', type=str, default='', help='eval data out dir')
def create_anno(result_path, dir_path):
    """
    create_label
    """
    file_list = glob.glob(dir_path + '/' + '*.jpg')
    file_list = sorted(file_list)

    with open(result_path + "/infer_anno.txt", "w") as f:
        for i in file_list:
            f.write(i.replace('.jpg', '').split('/')[-1] + '\n')

args = parser.parse_args()
if __name__ == "__main__":
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    create_anno(args.out_path, args.data_path)
