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
    """ change txt  """

    tofile = "after_update_" + file_path.split("/")[-1]
    fin = open(file_path, "r")
    fwrite = open(tofile, "w")
    for line in fin:
        imgAbsolutePath = line.split(" ")[0]

        print(line)
        print(
            "after update " + path + imgAbsolutePath.split("/")[-1] + " " + line.split(" ")[-3] + " " + line.split(" ")[
                -2] + " " + line.split(" ")[-1])
        fwrite.write(
            path + imgAbsolutePath.split("/")[-1] + " " + line.split(" ")[-3] + " " + line.split(" ")[-2] + " " +
            line.split(" ")[-1])
    fin.close()
    fwrite.close()


if __name__ == '__main__':
    # This parameter means the path to label.txt
    txt_path = sys.argv[1]
    dataset_path = sys.argv[2]
    change_zj_jk(txt_path, dataset_path)
