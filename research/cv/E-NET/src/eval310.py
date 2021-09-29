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
"""eval for 310 infer"""
import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image


def resize(img, height, interpolation):
    h, w = img.size
    width = int(height * h / w)
    img_new = img.resize((width, height), interpolation)
    return img_new


def load_image(file):
    return Image.open(file)


def is_label(filename):
    return filename.endswith("_labelTrainIds.png")


def convert_to_one_hot(y, C):
    return np.transpose(np.eye(C)[y], (0, 3, 1, 2)).astype(np.float32)


class iouEval:
    """compute iou"""

    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1
        self.reset()

    def reset(self):
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses-1
        self.tp = np.zeros(classes)
        self.fp = np.zeros(classes)
        self.fn = np.zeros(classes)

    def addBatch(self, x, y):
        """add a batch and compute its iou"""
        x_onehot = convert_to_one_hot(x, self.nClasses)
        y_onehot = convert_to_one_hot(y, self.nClasses)

        if self.ignoreIndex != -1:
            ignores = np.expand_dims(y_onehot[:, self.ignoreIndex], axis=1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0

        tpmult = x_onehot * y_onehot
        tp = np.sum(
            np.sum(np.sum(tpmult, axis=0, keepdims=True),
                   axis=2, keepdims=True),
            axis=3, keepdims=True
        ).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores)
        fp = np.sum(np.sum(np.sum(fpmult, axis=0,
                                  keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot)
        fn = np.sum(np.sum(np.sum(fnmult, axis=0,
                                  keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True).squeeze()

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return np.mean(iou), iou


class cityscapes_datapath:
    """cityscapes datapath iter"""
    def __init__(self, labels_path):

        self.labels_path = labels_path

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in
                            os.walk(os.path.expanduser(self.labels_path)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

    def __getitem__(self, index):
        filenameGt = self.filenamesGt[index]

        return filenameGt

    def __len__(self):
        return len(self.filenamesGt)




if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--res_path', type=str)
    parser.add_argument('--label_path', type=str)
    config = parser.parse_args()
    res_path = config.res_path
    label_path = config.label_path

    gt = {}

    for i in list(cityscapes_datapath(label_path)):
        gt[i.split("/")[-1].rstrip("_gtFine_labelTrainIds.png")] = i

    metrics = iouEval(nClasses=20)
    for i, bin_name in enumerate(os.listdir(res_path)):
        print(i)
        file_name_sof = os.path.join(res_path, bin_name)
        key = bin_name.split("_leftImg8bit_0.bin")[0]
        with open(gt[key], 'rb') as f:
            target = load_image(f).convert('P')
        target = resize(target, 512, Image.NEAREST)
        target = np.array(target).astype(np.uint32)
        target[target == 255] = 19
        target = target.reshape((1, 512, 1024))
        softmax_out = np.fromfile(
            file_name_sof, np.float32).reshape((1, 20, 512, 1024))
        preds = softmax_out.argmax(axis=1).astype(np.int32)
        labels = target.astype(np.int32)
        metrics.addBatch(preds, labels)

mean_iou, iou_class = metrics.getIoU()
mean_iou = mean_iou.item()

with open("metric.txt", "w") as metric_file:
    print("mean_iou: ", mean_iou, file=metric_file)
    print("iou_class: ", iou_class, file=metric_file)

print("mean_iou: ", mean_iou)
print("iou_class: ", iou_class)
