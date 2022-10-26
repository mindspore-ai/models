# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
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
"""the main sdk infer file"""
import argparse
import os
import cv2
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser('mindspore PSPNet eval')
    parser.add_argument('--data_lst', type=str, default='', help='list of val data')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='number of classes')
    parser.add_argument('--result_path', type=str, default='./result',
                        help='the result path')
    parser.add_argument('--name_txt', type=str,
                        default='',
                        help='the name_txt path')
    args, _ = parser.parse_known_args()
    return args


def intersectionAndUnion(output, target, K, ignore_index=255):
    """
    'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    """

    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    print("output.shape=", output.shape)
    print("output.size=", output.size)
    output = output.reshape(output.size).copy()  # output= [0 0 0 ... 0 0 0]
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]

    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))

    # IoU = A+B -AnB
    area_union = area_output + area_target - area_intersection  # [107407 0 0 0 0 153165 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    return area_intersection, area_union, area_target


def cal_acc(data_list, pred_folder, classes, names):
    """ Calculation evaluating indicator """
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    with open(data_list) as f:
        img_lst = f.readlines()
        for i, line in enumerate(img_lst):
            image_path, target_path = line.strip().split(' ')
            image_name = image_path.split('/')[-1].split('.')[0]
            pred = cv2.imread(os.path.join(pred_folder, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
            target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            intersection, union, target = intersectionAndUnion(pred, target, classes)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            print('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(
                i + 1, len(data_list), image_name + '.png', accuracy))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)  # 计算所有类别交集和并集之比的平均值
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(classes):
            print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(
                i, iou_class[i], accuracy_class[i], names[i]))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        """ calculate the result """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = _parse_args()
    gray_folder = os.path.join(args.result_path, "gray")
    names = [line.rstrip('\n') for line in open(args.name_txt)]
    cal_acc(args.data_lst, gray_folder, args.num_classes, names)


if __name__ == '__main__':
    main()
