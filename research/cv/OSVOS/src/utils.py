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
"""
util functions.
"""
import os
import random
import cv2
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from sklearn.metrics import jaccard_score, f1_score


class ScaleNRotate():
    """
    Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert isinstance(rots, type(scales))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        if isinstance(self.rots, tuple):
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2
            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif isinstance(self.rots, list):
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            tmp = sample[elem]
            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert center != 0
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample


class RandomHorizontalFlip():
    """
    Horizontally flip the given image and ground truth randomly.
    """
    def __call__(self, sample):
        if random.random() < 0.5:
            for elem in sample.keys():
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp
        return sample


class ClassBalancedCrossEntropyLoss(nn.Cell):
    """
    the definition of balanced CrossEntropyLoss.
    """
    def __init__(self, size_average=True, batch_average=True, online=False):
        super(ClassBalancedCrossEntropyLoss, self).__init__()

        self.size_average = size_average
        self.batch_average = batch_average
        self.online = online
        self.sum = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.log = P.Log()
        self.exp = P.Exp()
        self.expand_dims = P.ExpandDims()

    def construct(self, outputs, label):
        label = self.expand_dims(label, 1)
        total_loss = None

        if self.online:
            output = outputs[-1]
            labels = (label >= 0.5).astype(mindspore.float32)

            num_labels_pos = self.sum(labels)
            num_labels_neg = self.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg

            output_gt_zero = (output >= 0).astype(mindspore.float32)
            loss_val = self.mul(output, (labels - output_gt_zero)) - self.log(
                1 + self.exp(output - 2 * self.mul(output, output_gt_zero)))

            loss_pos = self.sum(-self.mul(labels, loss_val))
            loss_neg = self.sum(-self.mul(1.0 - labels, loss_val))

            final_loss = num_labels_neg / num_total * \
                         loss_pos + num_labels_pos / num_total * loss_neg

            if self.size_average:
                final_loss /= (label.shape[0] *
                               label.shape[1] * label.shape[2] * label.shape[3])
            elif self.batch_average:
                final_loss /= label.shape[0]

            if total_loss is None:
                total_loss = final_loss
            else:
                total_loss = total_loss + final_loss

            return total_loss

        for i in range(len(outputs)):
            output = outputs[i]

            labels = (label >= 0.5).astype(mindspore.float32)

            num_labels_pos = self.sum(labels)
            num_labels_neg = self.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg

            output_gt_zero = (output >= 0).astype(mindspore.float32)
            loss_val = self.mul(output, (labels - output_gt_zero)) - self.log(
                1 + self.exp(output - 2 * self.mul(output, output_gt_zero)))

            loss_pos = self.sum(-self.mul(labels, loss_val))
            loss_neg = self.sum(-self.mul(1.0 - labels, loss_val))

            final_loss = num_labels_neg / num_total * \
                         loss_pos + num_labels_pos / num_total * loss_neg

            if self.size_average:
                final_loss /= (label.shape[0] *
                               label.shape[1] * label.shape[2] * label.shape[3])
            elif self.batch_average:
                final_loss /= label.shape[0]

            if i < len(outputs) - 1:
                final_loss = 0.5 * final_loss

            if total_loss is None:
                total_loss = final_loss
            else:
                total_loss = total_loss + final_loss

        return total_loss


class Evaluation():
    """
    the eval function for the prediction images.
    """
    def __init__(self, eval_txt, pred_path, gt_path):
        self.eval_txt = eval_txt
        self.pred_path = pred_path
        self.gt_path = gt_path
        self.metrics_J = {}
        self.metrics_F = {}

    def evaluate(self):
        with open(self.eval_txt) as f:
            seqs = f.readlines()
        for seq in seqs:
            seq = seq.strip()
            self.metrics_J[str(seq)] = []
            self.metrics_F[str(seq)] = []

            pred_dir = os.path.join(self.pred_path, seq)
            gt_dir = os.path.join(self.gt_path, seq)
            preds_name = np.sort(os.listdir(pred_dir))
            preds = []
            gts = []
            for name in preds_name:
                pred = np.array(cv2.imread(os.path.join(pred_dir, name), 0))
                pred[pred > 50] = 255
                pred[pred <= 50] = 0
                gt = np.array(cv2.imread(os.path.join(gt_dir, name), 0))
                preds.append(pred)
                gts.append(gt)
            preds = np.reshape(np.array(preds)/255, (-1,))
            gts = np.reshape(np.array(gts)/255, (-1,))

            self.metrics_J[str(seq)].append(jaccard_score(preds, gts, average='binary'))
            self.metrics_F[str(seq)].append(f1_score(preds, gts, average='binary'))

        return self.metrics_J, self.metrics_F
