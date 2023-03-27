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
"""evaluation metric."""
import numpy as np

from mindspore import nn

import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore._checkparam import Validator as validator


classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
           'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
           'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
           'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cls in classes:
    for label in classes[cls]:
        label_to_cat[label] = cls

class WithEvalCell(nn.Cell):
    def __init__(self, network, add_cast_fp32=False):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self._log_softmax = nn.LogSoftmax()
        self.cast = P.Cast()
        self.add_cast_fp32 = validator.check_value_type("add_cast_fp32", add_cast_fp32, [bool], self.cls_name)

    def construct(self, data, label1, label2):
        outputs = self._network(data)
        if self.add_cast_fp32:
            label1 = F.mixed_precision_cast(mstype.float32, label1)
            label2 = F.mixed_precision_cast(mstype.float32, label2)
            outputs = self.cast(outputs, mstype.float32)

        pred = self._log_softmax(outputs)

        return  pred, label2

class IoU(nn.Metric):
    def __init__(self):
        super(IoU, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self.shape_ious = {cat: [] for cat in classes}

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Distribute accuracy needs 2 input (y_correct), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        target = self._convert_data(inputs[1])
        B, N, _ = y_pred.shape
        cur_pred_val_logits = y_pred
        cur_pred_val = np.zeros((B, N)).astype(np.int32)

        for i in range(B):
            cat = label_to_cat[target[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cat_logits = logits[:, classes[cat]]
            cur_pred_val[i, :] = np.argmax(cat_logits, 1) + classes[cat][0]

        for i in range(B):
            pred_seg = cur_pred_val[i, :]
            leb_seg = target[i, :]
            cat = label_to_cat[leb_seg[0]]
            part_ious = [0.0 for _ in range(len(classes[cat]))]
            for category in classes[cat]:
                if (np.sum(leb_seg == category) == 0) and (
                        np.sum(pred_seg == category) == 0):
                    part_ious[category - classes[cat][0]] = 1.0
                else:
                    part_ious[category - classes[cat][0]] = np.sum((leb_seg == category) & (pred_seg == category)) / \
                                                       float(np.sum((leb_seg == category) | (pred_seg == category)))
            self.shape_ious[cat].append(np.mean(part_ious))


    def eval(self):
        all_shape_ious = []
        class_shape_ious = {}
        for category in self.shape_ious:
            for iou in self.shape_ious[category]:
                all_shape_ious.append(iou)
            class_shape_ious[category] = np.mean(self.shape_ious[category])

        class_avg_iou = np.mean(list(class_shape_ious.values()))
        inctance_avg_iou = np.mean(all_shape_ious)

        return class_avg_iou, inctance_avg_iou
