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


from argparse import ArgumentParser


def parse_args():
    """
    parse args
    """
    parser = ArgumentParser(description="generate train_list.txt or val_list.txt")
    parser.add_argument("--dataset_list_txt", type=str, default="", help="path of val.txt in VOC2012")
    parser.add_argument("--image_prefix", type=str, default="JPEGImages",
                        help="the relative path in the data_root, until to the level of images.jpg")
    parser.add_argument("--mask_prefix", type=str, default="SegmentationClass",
                        help="the relative path in the data_root, until to the level of mask.jpg")
    parser.add_argument("--output_txt", type=str, default="voc2012_val.txt", help="name of output txt")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.dataset_list_txt, 'r') as f:
        with open(args.output_txt, 'w') as fw:
            for i in f:
                image_path = args.image_prefix + "/" + i[:-1] + ".jpg "
                label_path = args.mask_prefix + "/" + i[:-1] + ".png\n"
                line = image_path + label_path
                fw.writelines(line)
            fw.close()
        f.close()

main()
