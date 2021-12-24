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
"""Self-defined callbacks."""
import os
import time
import numpy as np

import mindspore.ops as ops
from mindspore.train.callback import Callback
from mindspore import save_checkpoint
from mindspore import Tensor


class SegEvalCallback(Callback):
    """Callback for inference while training. Dataset cityscapes."""
    def __init__(self, loader, net, num_classes=19, ignore_label=255,
                 start_epoch=0, save_path=None, interval=1):
        super(SegEvalCallback, self).__init__()
        self.loader = loader
        self.net = net
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.start_epoch = start_epoch
        self.save_path = save_path
        self.interval = interval
        self.best_miou = 0

    def epoch_end(self, run_context):
        """Epoch end."""
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch >= self.start_epoch:
            if (cur_epoch - self.start_epoch) % self.interval == 0:
                self.net.set_train(False)
                device_id = int(os.getenv("DEVICE_ID"))
                miou = self.inference()
                if miou > self.best_miou:
                    self.best_miou = miou
                    if self.save_path:
                        file_path = os.path.join(self.save_path, f"best-{device_id}.ckpt")
                        save_checkpoint(self.net, file_path)
                print("=== epoch: {:4d}, device id: {:2d}, best miou: {:6.4f}, miou: {:6.4f}".format(
                    cur_epoch, device_id, self.best_miou, miou), flush=True)
                self.net.set_train(True)

    def inference(self):
        """Cityscapes inference."""
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for batch in self.loader:
            image, label = batch
            shape = label.shape
            pred = self.net(image)
            pred = ops.ResizeBilinear((shape[-2], shape[-1]))(pred)
            pred = ops.Exp()(pred)

            confusion_matrix += self.get_confusion_matrix(label, pred, shape,
                                                          self.num_classes, self.ignore_label)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        iou_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_iou = iou_array.mean()
        return mean_iou

    def get_confusion_matrix(self, label, pred, shape, num_class, ignore=255):
        """
        Calcute the confusion matrix by given label and pred.
        """
        output = pred.asnumpy().transpose(0, 2, 3, 1)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        seg_gt = np.asarray(label.asnumpy()[:, :shape[-2], :shape[-1]], dtype=np.int32)

        ignore_index = seg_gt != ignore
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]

        index = (seg_gt * num_class + seg_pred).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((num_class, num_class))

        for i_label in range(num_class):
            for i_pred in range(num_class):
                cur_index = i_label * num_class + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred] = label_count[cur_index]
        return confusion_matrix


class TimeLossMonitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> TimeLossMonitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(TimeLossMonitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)
        self.losses = []
        self.epoch_time = 0
        self.step_time = 0

    def epoch_begin(self, run_context):
        """Epoch begin."""
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """Epoch end."""
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch: [{:3d}/{:3d}], epoch time: {:5.3f}, steps: {:5d}, "
              "per step time: {:5.3f}, avg loss: {:5.3f}, lr:[{:8.6f}]".format(
                  cb_params.cur_epoch_num, cb_params.epoch_num, epoch_mseconds, cb_params.batch_num,
                  per_step_mseconds, np.mean(self.losses), self.lr_init[cb_params.cur_step_num - 1]), flush=True)

    def step_begin(self, run_context):
        """Step begin."""
        self.step_time = time.time()

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
