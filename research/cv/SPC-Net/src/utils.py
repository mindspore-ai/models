# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import numpy as np
from PIL import Image

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
ZERO_PAD = 256 * 3 - len(palette)
for i in range(ZERO_PAD):
    palette.append(0)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def save_imgs(prediction=None, img_name='default', save_path='./'):
    col_img_name = os.path.join(save_path, '{}.png'.format(img_name))

    colorized = colorize_mask(prediction)
    colorized.save(col_img_name)


def fast_hist(label_pred, label_true, num_classes=19):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def print_evaluate_results(hist, iu, dataset_name='', num_classes=19):
    # fixme: Need to refactor this dict
    id2cat = {i: i for i in range(num_classes)}
    iu_false_positive = hist.sum(axis=1) - np.diag(hist)
    iu_false_negative = hist.sum(axis=0) - np.diag(hist)
    iu_true_positive = np.diag(hist)

    print('Dataset name: {}'.format(dataset_name))
    print('IoU:')
    print('label_id       label    iU    Precision Recall  TP      FP      FN')
    for idx, itera in enumerate(iu):
        # Format all of the strings:
        idx_string = "{:2d}".format(idx)
        class_name = "{:>13}".format(id2cat[idx]) if idx in id2cat else ''
        iu_string = '{:5.1f}'.format(itera * 100)
        total_pixels = hist.sum()
        tp = '{:5.1f}'.format(100 * iu_true_positive[idx] / total_pixels)
        fp = '{:5.1f}'.format(
            iu_false_positive[idx] / iu_true_positive[idx])
        fn = '{:5.1f}'.format(iu_false_negative[idx] / iu_true_positive[idx])
        precision = '{:5.1f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx]))
        recall = '{:5.1f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx]))
        print('{}    {}   {}  {}     {}  {}   {}   {}'.format(
            idx_string, class_name, iu_string, precision, recall, tp, fp, fn))


def evaluate_eval_for_inference(hist, dataset_name='', num_classes=19):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    """
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    print_evaluate_results(hist, iu, dataset_name=dataset_name, num_classes=num_classes)
    freq = hist.sum(axis=1) / hist.sum()
    mean_iu = np.nanmean(iu)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return {
        "acc": acc,
        "acc_cls": acc_cls,
        "mean_iu": mean_iu,
        "fwavacc": fwavacc
    }
