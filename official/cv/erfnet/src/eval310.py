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
import os
from argparse import ArgumentParser
import numpy as np
import torch
from torchvision.transforms import Resize
from PIL import Image

def load_image(fileName):
    return Image.open(fileName)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class iouEval_1:

    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1
        self.reset()

    def reset(self):
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()

    def addBatch(self, x, y):   #x=preds, y=targets
        #sizes should be "batch_size x nClasses x H x W"

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        if x.size(1) == 1:
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if y.size(1) == 1:
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if self.ignoreIndex != -1:
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0

        tpmult = x_onehot * y_onehot
        tp = torch.sum(
            torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True),
            dim=3, keepdim=True
        ).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, \
            keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot)
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, \
            keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou

class cityscapes_datapath:

    def __init__(self, labels_path):

        self.labels_path = labels_path

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in \
            os.walk(os.path.expanduser(self.labels_path)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

    def __getitem__(self, index):
        filenameGt = self.filenamesGt[index]

        return filenameGt

    def __len__(self):
        return len(self.filenamesGt)

# example:
# python eval.py --res_path \
# --label_path

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

    metrics = iouEval_1(nClasses=20)
    for i, bin_name in enumerate(os.listdir(res_path)):
        print(i)
        file_name_sof = os.path.join(res_path, bin_name)
        key = bin_name.split("_leftImg8bit_0.bin")[0]
        with open(gt[key], 'rb') as f:
            target = load_image(f).convert('P')
        target = Resize(512, Image.NEAREST)(target)
        target = np.array(target).astype(np.uint32)
        target[target == 255] = 19
        target = target.reshape(512, 1024)
        target = target[np.newaxis, :, :]
        softmax_out = np.fromfile(file_name_sof, np.float32)
        softmax_out = softmax_out.reshape(1, 20, 512, 1024)
        preds = torch.Tensor(softmax_out.argmax(axis=1).astype(np.int32)).unsqueeze(1).long()
        labels = torch.Tensor(target.astype(np.int32)).unsqueeze(1).long()
        metrics.addBatch(preds, labels)

mean_iou, iou_class = metrics.getIoU()
mean_iou = mean_iou.item()

with open("metric.txt", "w") as file:
    print("mean_iou: ", mean_iou, file=file)
    print("iou_class: ", iou_class, file=file)

print("mean_iou: ", mean_iou)
print("iou_class: ", iou_class)
