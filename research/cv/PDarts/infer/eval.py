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
""" eval_sdk.py """
import os
import argparse
import numpy as np

def parse_args(parsers):
    """
    Parse commandline arguments.
    """
    parsers.add_argument('--images_txt_path', type=str, default="./data/preprocess_Result/label.txt",
                         help='image path')
    parsers.add_argument('--infer_results_txt', type=str, default="./mxbase/build/infer_results.txt")
    parsers.add_argument('--result_log_path', type=str, default="./results/eval_mxbase.log")
    return parsers

def read_file_list(input_f):
    """
    :param infer file content:
        1.bin 0
        2.bin 2
        ...
    :return image path list, label list
    """
    image_file_l = []
    label_l = []
    if not os.path.exists(input_f):
        print('input file does not exists.')
    with open(input_f, "r") as fs:
        for line in fs.readlines():
            line = line.strip('\n').split(',')
            file = line[0]
            label = int(line[1])
            image_file_l.append(file)
            label_l.append(label)
    return image_file_l, label_l

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Om pdarts Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    images_txt_path = args.images_txt_path
    infer_results_txt = args.infer_results_txt
    # load results and label
    results = np.loadtxt(infer_results_txt)
    file_list, label_list = read_file_list(images_txt_path)
    img_size = len(file_list)

    labels = np.array(label_list)
    # cal acc
    acc_top1 = (results[:, 0] == labels).sum() / img_size
    print('Top1 acc:', acc_top1)

    acc_top5 = sum([1 for i in range(img_size) if labels[i] in results[i]]) / img_size
    print('Top5 acc:', acc_top5)

    with open(args.result_log_path, 'w') as f:
        f.write('Eval size: {} \n'.format(img_size))
        f.write('Top1 acc: {} \n'.format(acc_top1))
        f.write('Top5 acc: {} \n'.format(acc_top5))
