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
"""eval process"""
import os
import argparse
import ast

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type=str, help='pred_dir, default: None')
parser.add_argument("--label_dir", type=str, help='label_dir, default: None')

# additional params for online evaluating
parser.add_argument("--run_online", type=ast.literal_eval, default=False, help='whether train online, default: false')
parser.add_argument("--data_url", type=str, help='path to data on obs, default: None')
parser.add_argument("--train_url", type=str, help='output path on obs, default: None')

args = parser.parse_args()
if __name__ == '__main__':
    if args.run_online:
        import moxing as mox
        mox.file.copy_parallel(args.data_url, "/cache/dataset")
        pred_dir = "/cache/dataset/pred_dir"
        label_dir = "/cache/dataset/label_dir"

    else:
        pred_dir = args.pred_dir
        label_dir = args.label_dir

    def generate(pred, label, num=255):
        """generate prec and recall"""
        prec = np.zeros(num)
        recall = np.zeros(num)
        min_num = 0
        max_num = 1 - 1e-10
        for i in range(num):
            tmp_num = (max_num - min_num) / num * i
            pred_ones = pred >= tmp_num
            acc = (pred_ones * label).sum()
            recall[i] = acc / (pred_ones.sum() + 1e-20)
            prec[i] = acc / (label.sum() + 1e-20)
        return prec, recall


    def F_score(pred, label, num=255):
        """calculate f-score"""
        prec, recall = generate(pred, label, num)
        beta2 = 0.3
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall + 1e-20)
        return f_score


    def max_F(pred_directory, label_directory, num=255):
        """calculate max f-score"""
        sum_value = np.zeros(num)
        content_list = os.listdir(pred_directory)
        pic_num = 0
        for i in range(len(content_list)):

            pred_path = pred_directory + "/" + content_list[i]
            pred = np.array(Image.open(pred_path), dtype='float32')

            pic_name = content_list[i].replace(".jpg", "").replace(".png", "").replace(".JPEG", "")
            print("%d / %d ,  %s \n" % (i+1, len(content_list), pic_name))
            label_path = os.path.join(label_directory, pic_name) + ".png"
            label = np.array(Image.open(label_path), dtype='float32')
            if len(label.shape) > 2:
                label = label[:, :, 0]

            if len(pred.shape) > 2:
                print(pred.shape)
                pred = pred[:, :, 0]
                pred = pred.squeeze()
                print(pred.shape)
            label /= label.max()
            pred /= pred.max()
            tmp = F_score(pred, label, num)
            sum_value += tmp
            pic_num += 1
        score = sum_value / pic_num
        return score.max()

    print("max_F measure, score = %f" % (max_F(pred_dir, label_dir)))
