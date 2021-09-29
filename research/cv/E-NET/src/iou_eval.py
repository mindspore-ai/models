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
"""compute iou"""
import numpy as np

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
            np.sum(np.sum(tpmult, axis=0, keepdims=True), axis=2, keepdims=True),
            axis=3, keepdims=True
        ).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores)
        fp = np.sum(np.sum(np.sum(fpmult, axis=0, \
            keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot)
        fn = np.sum(np.sum(np.sum(fnmult, axis=0, \
            keepdims=True), axis=2, keepdims=True), axis=3, keepdims=True).squeeze()

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return np.mean(iou), iou
