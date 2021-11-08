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
"""
generate accuracy through 310
"""
import os
import argparse
import numpy as np

from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser('mindspore tsn testing')
# Path for data
parser.add_argument('--dataset', type=str, default="ucf101", choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('--test_list', type=str, default="")
parser.add_argument('--modality', type=str, default="RGB", choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('--save_scores', type=str, default="./score/score_", help="./score/score_flow_2_")
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--label_dir', type=str, default='', help='label data directory.')
parser.add_argument('--result_dir', type=str, default="./result_Files", help='infer result dir.')

args = parser.parse_args()

if __name__ == "__main__":

    rst_path = args.result_dir
    labels = np.load(args.label_dir)
    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    outputs = []

    num_crop = args.test_crops

    for i in range(len(os.listdir(rst_path))):
        file_name = os.path.join(rst_path, "TSN_data_bs" + str(1) + '_' + str(i) + '_0.bin')
        output = np.fromfile(file_name, np.float16)

        rst = output.copy()

        rst = rst.reshape((num_crop, args.test_segments,\
         num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

        outputs.append((rst, labels[i].asnumpy().tolist()))

    video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in outputs]
    video_labels = [x[1] for x in outputs]

    cf = confusion_matrix(video_labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Accuracy {:.01f}%'.format(np.mean(cls_acc) * 100))

    if args.save_scores is not None:

        # reorder before saving
        name_list = [x.strip().split()[0] for x in open(args.test_list)]
        order_dict = {e: i for i, e in enumerate(sorted(name_list))}

        reorder_output = [None] * len(outputs)
        reorder_label = [None] * len(outputs)

        for i in range(len(outputs)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = outputs[i]
            reorder_label[idx] = video_labels[i]

        np.savez(args.save_scores+args.modality, scores=reorder_output, labels=reorder_label)
