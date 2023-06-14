# coding = utf-8
"""
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


import sys


def change_zj_jk(file_path, path):
    """change txt"""

    tofile = "after_update_" + file_path.split("/")[-1]
    fin = open(file_path, "r")
    fwrite = open(tofile, "w")
    for line in fin:
        identity = line.split(" ")[1]
        pathimg = line.split(" ")[0].split("/")

        print(line)
        print("修改过后的" + path + pathimg[-3] + "/" + pathimg[-2] + "/" + pathimg[-1])
        fwrite.write(path + pathimg[-3] + "/" + pathimg[-2] + "/" + pathimg[-1] + " " + identity)
    fin.close()
    fwrite.close()


def change_dis(file_path, path):
    """change txt"""

    tofile = "./after_update_dis_list.txt"
    fin = open(file_path, "r")
    fwrite = open(tofile, "w")
    for line in fin:
        pathimg = line.split("/")

        print(line)
        print("修改过后 " + path + pathimg[-3] + "/" + pathimg[-2] + "/" + pathimg[-1])
        fwrite.write(path + pathimg[-3] + "/" + pathimg[-2] + "/" + pathimg[-1])
    fin.close()
    fwrite.close()


if __name__ == "__main__":
    # This parameter means the path to zj_lists.txt
    zj_txt_path = sys.argv[1]
    # This parameter means the path to jk_lists.txt
    jk_txt_path = sys.argv[2]
    # This parameter means the path to dis_lists.txt
    dis_txt_path = sys.argv[3]
    # This parameter means the path to dataset path
    dataset_path = sys.argv[4]
    change_zj_jk(zj_txt_path, dataset_path)
    change_zj_jk(jk_txt_path, dataset_path)
    change_dis(dis_txt_path, dataset_path)
