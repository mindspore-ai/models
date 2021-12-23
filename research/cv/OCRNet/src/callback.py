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

"""Callback for inference while training."""
import os
import mindspore.ops as P
from mindspore.train.callback import Callback
from mindspore import save_checkpoint
import numpy as np


def get_confusion_matrix(label, pred, shape, num_classes, ignore_label):
    """Calcute the confusion matrix by given label and pred."""
    output = pred.transpose(0, 2, 3, 1)  # NHWC
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)  # NHW
    seg_gt = np.asarray(label[:, :shape[-2], :shape[-1]], dtype=np.int32)  # NHW

    ignore_index = seg_gt != ignore_label  # NHW
    seg_gt = seg_gt[ignore_index]  # NHW
    seg_pred = seg_pred[ignore_index]  # NHW

    index = (seg_gt * num_classes + seg_pred).astype(np.int32)
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred in range(num_classes):
            cur_index = i_label * num_classes + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def evaluate_model(net, data_helper, num_classes, ignore_label):
    """Inference function."""
    net.set_train(False)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for item in data_helper:
        image = item[0]
        label = item[1].asnumpy()
        shape = label.shape
        pred = net(image)
        pred = pred[-1]
        pred = P.ResizeBilinear((shape[-2], shape[-1]))(pred).asnumpy()
        confusion_matrix += get_confusion_matrix(label, pred, shape, num_classes, ignore_label)
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    return IoU_array, mean_IoU


class EvalCallback(Callback):
    """Callback for inference while training."""
    def __init__(self, network, eval_data, num_classes, ignore_label, train_url, eval_interval=1):
        self.network = network
        self.eval_data = eval_data
        self.best_iouarray = None
        self.best_miou = 0
        self.best_epoch = 0
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.eval_interval = eval_interval
        self.train_url = train_url

    def epoch_end(self, run_context):
        """Executions after each epoch."""
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        device_id = int(os.getenv("DEVICE_ID"))
        if cur_epoch % self.eval_interval == 0:
            iou_array, miou = evaluate_model(self.network, self.eval_data, self.num_classes, self.ignore_label)
            if miou > self.best_miou:
                self.best_miou = miou
                self.best_iouarray = iou_array
                self.best_epoch = cur_epoch
                save_checkpoint(self.network, self.train_url + "/best_card%d.ckpt" % device_id)

            log_text1 = 'EPOCH: %d, mIoU: %.4f\n' % (cur_epoch, miou)
            log_text2 = 'BEST EPOCH: %s, BEST mIoU: %0.4f\n' % (self.best_epoch, self.best_miou)
            log_text3 = 'DEVICE_ID: %d\n' % device_id
            print("==================================================\n",
                  log_text3,
                  log_text1,
                  log_text2,
                  "==================================================")
            self.network.set_train(True)
