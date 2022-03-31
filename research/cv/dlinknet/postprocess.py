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

import sys
import os
import numpy as np
import cv2


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        _iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        _iou = _iou[0]
        # mean acc
        _acc = np.diag(self.hist).sum() / self.hist.sum()
        _acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        return _acc, _acc_cls, _iou


def tranform_bin_to_png(_predictpath):
    _pres = os.listdir(_predictpath)
    for file in _pres:
        if file[-4:] == '.bin':
            _pre = np.fromfile(os.path.join(_predictpath, file), np.float32).reshape(1024, 1024)
            _pre[_pre > 0.5] = 255
            _pre[_pre <= 0.5] = 0
            _pre = np.concatenate([_pre[:, :, None], _pre[:, :, None], _pre[:, :, None]], axis=2)
            file = file.split('.')[0]
            file = file.split('_')[0] + "_mask.png"
            cv2.imwrite(os.path.join(_predictpath, file), _pre.astype(np.uint8))


if __name__ == '__main__':
    predictpath = sys.argv[1]
    tranform_bin_to_png(predictpath)
    label_path = sys.argv[2]
    pres = os.listdir(predictpath)
    labels = []
    predicts = []
    for im in pres:
        if im[-4:] == '.png':
            lab_path = os.path.join(label_path, im)
            pre_path = os.path.join(predictpath, im)
            label = cv2.imread(lab_path, 0)
            pre = cv2.imread(pre_path, 0)
            label[label > 0] = 1
            pre[pre > 0] = 1
            labels.append(label)
            predicts.append(pre)
    el = IOUMetric(2)
    acc, acc_cls, iou = el.evaluate(predicts, labels)
    print('acc: ', acc)
    print('acc_cls: ', acc_cls)
    print('iou: ', iou)
