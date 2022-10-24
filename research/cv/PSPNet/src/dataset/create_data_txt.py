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
"""generate train_list.txt or val_list.txt file script"""
import os
from argparse import ArgumentParser

def parse_args():
    """
    parse args
    """
    parser = ArgumentParser(description="generate train_list.txt or val_list.txt")
    parser.add_argument("--data_root", type=str, default="", help="data root path, should be same as data_root in yaml")
    parser.add_argument("--image_prefix", type=str, default="", help="the relative path in the data_root, "
                                                                     "until to the level of images.jpg")
    parser.add_argument("--mask_prefix", type=str, default="", help="the relative path in the data_root, "
                                                                    "until to the level of mask.jpg")
    parser.add_argument("--output_txt", type=str, default="", help="name of output txt")
    args = parser.parse_args()
    return args

def findAllFile(base):
    """
    Recursive search all of the files under base path
    """
    for root, _, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def combine_txt(txt1, txt2, output_txt):
    """
    combine txt1 and txt2
    """
    with open(txt1, 'r') as fa:
        with open(txt2, 'r') as fb:
            with open(output_txt, 'w') as fc:
                for line in fa:
                    fc.write(line.strip('\r\n'))
                    fc.write(' ')
                    fc.write(fb.readline())

def sort_txt(txt, sorted_txt):
    """
    sort txt
    """
    names = []
    with open(txt, 'r') as f:
        for line in f:
            names.append(line.strip())
    with open(sorted_txt, 'w') as f:
        for item in sorted(names):
            f.writelines(item)
            f.writelines('\n')
        f.close()

def main():
    """
    get image name and mask name and write in the output_txt
    """
    args = parse_args()
    image_txt = "image.txt"
    mask_txt = "mask.txt"
    image_sort_txt = "image_sort.txt"
    mask_sort_txt = "mask_sort.txt"
    f1 = open(image_txt, 'a')
    f2 = open(mask_txt, 'a')
    absolute_image_path = os.path.join(args.data_root, args.image_prefix)
    absolute_mask_path = os.path.join(args.data_root, args.mask_prefix)
    print("absolute_image_path is:", absolute_image_path, " absolute_mask_path is:", absolute_mask_path)
    assert absolute_image_path is not None, "absolute image path is None."
    assert absolute_mask_path is not None, "absolute mask path is Node."
    for i in findAllFile(absolute_image_path):
        a = i.replace(args.data_root, '')
        f1.write(a)
        f1.write("\n")
    f1.close()
    for j in findAllFile(absolute_mask_path):
        b = j.replace(args.data_root, '')
        f2.write(b)
        f2.write("\n")
    f2.close()
    sort_txt(image_txt, image_sort_txt)
    sort_txt(mask_txt, mask_sort_txt)
    combine_txt(image_sort_txt, mask_sort_txt, args.output_txt)
    os.remove(image_txt)
    os.remove(mask_txt)
    os.remove(image_sort_txt)
    os.remove(mask_sort_txt)

if __name__ == "__main__":
    main()
