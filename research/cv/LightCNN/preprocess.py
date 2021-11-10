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
import argparse
from src.dataset import create_dataset
parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--data_path', type=str, required=True, help='eval data dir')
args = parser.parse_args()


def edit_blufr_image_list(list_file_path):
    """edit image list for Ascend310 blufr test"""
    f = open(list_file_path, 'r')
    new_file_path = args.data_path + "/image_list_for_blufr_310test.txt"
    f1 = open(new_file_path, 'w')
    for line in f:
        img_name = line[:-4]
        person_name = line[:img_name.rfind('_')]
        path = person_name + '/' + img_name + 'bmp'
        line = path + ' 0'
        f1.write(line+'\n')
    return new_file_path


if __name__ == '__main__':
    lfw_list = args.data_path + "/image_list_for_lfw.txt"
    blufr_list = args.data_path + "/image_list_for_blufr.txt"
    data_path = args.data_path + "/image"
    result_path = args.data_path + "/bin_data"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    lfw_dataset = create_dataset(mode='Val', data_url=data_path, data_list=lfw_list,
                                 batch_size=1, num_of_workers=8)
    for idx, data in enumerate(lfw_dataset.create_dict_iterator()):
        img_data = data["image"]
        file_name = "lfw_" + str(idx).zfill(5) + ".bin"
        img_file_path = os.path.join(result_path, file_name)
        img_data.asnumpy().tofile(img_file_path)

    print("=" * 20, "export bin files for lfw 6,000 pairs test finished", "=" * 20)


    blufr_list_310test = edit_blufr_image_list(blufr_list)
    blufr_dataset = create_dataset(mode='Val', data_url=data_path, data_list=blufr_list_310test,
                                   batch_size=1, num_of_workers=8)
    for idx, data in enumerate(blufr_dataset.create_dict_iterator()):
        img_data = data["image"]
        file_name = "blufr_" + str(idx).zfill(5) + ".bin"
        img_file_path = os.path.join(result_path, file_name)
        img_data.asnumpy().tofile(img_file_path)

    print("=" * 20, "export bin files for lfw BLUFR protocols test finished", "=" * 20)
