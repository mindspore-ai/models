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
"""
cpu_cut_data.
"""
import os
import shutil


def generate_data():
    dirs = []
    path = "./"
    abs_path = None
    for abs_path, j, _ in os.walk(path):
        print("abs_path:", abs_path)
        if len(j).__trunc__() > 0:
            dirs.append(j)
    print(dirs)

    train_folder = os.path.exists("./train")
    if not train_folder:
        os.makedirs("./train")
    test_folder = os.path.exists("./test")
    if not test_folder:
        os.makedirs("./test")

    for di in dirs[0]:
        files = os.listdir(di)
        train_set = files[: int(len(files) * 3 / 4)]
        test_set = files[int(len(files) * 3 / 4):]
        for file in train_set:
            file_path = "./train/" + di + "/"
            folder = os.path.exists(file_path)
            if not folder:
                os.makedirs(file_path)
            src_file = "./" + di + "/" + file
            print("src_file:", src_file)
            dst_file = file_path + file
            print("dst_file:", dst_file)
            shutil.copyfile(src_file, dst_file)

        for file in test_set:
            file_path = "./test/" + di + "/"
            folder = os.path.exists(file_path)
            if not folder:
                os.makedirs(file_path)
            src_file = "./" + di + "/" + file
            dst_file = file_path + file
            shutil.copyfile(src_file, dst_file)


if __name__ == '__main__':
    generate_data()
