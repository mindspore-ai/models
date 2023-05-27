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
"""Util class or function."""

import os
import stat
from datetime import datetime
import numpy as np

from mindspore import nn
from mindspore import save_checkpoint
from mindspore import log as logger
from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor
from src.score import batch_pix_accuracy, batch_intersection_union

class TempLoss(nn.Cell):
    """A temp loss cell."""
    def construct(self, *inputs, **kwargs):
        return 0.1

class SegmentationMetric(nn.Metric):
    """FastSCNN Metric, computes pixAcc and mIoU metric scores."""
    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.clear()

    def clear(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.zeros(self.nclass)
        self.total_union = np.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0

    def update(self, *inputs):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
        preds, labels = inputs[0], inputs[-1]
        preds = preds[0]
        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred.asnumpy(), label.asnumpy())
            inter, union = batch_intersection_union(pred.asnumpy(), label.asnumpy(), self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def eval(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        # remove np.spacing(1)
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU


class EvalCallBack(Callback):
    """
    Evaluation callback when training.

    Args:
        network (function): evaluation network.
        dataloader (dict): evaluation dataloader.
        interval (int): run evaluation interval, default is 1.
        eval_start_epoch (int): evaluation start epoch, default is 1.
        save_best_ckpt (bool): Whether to save best checkpoint, default is True.
        best_ckpt_name (str): best checkpoint name, default is `best.ckpt`.
        metrics_name (str): evaluation metrics name, default is ("pixAcc", "mIou").

    Returns:
        None

    Examples:
        >>> EvalCallBack(network, dataloader)
    """

    def __init__(self, network, dataloader, interval=1, eval_start_epoch=1, \
        save_best_ckpt=True, ckpt_directory="./", best_ckpt_name="best.ckpt", metrics_name=("pixAcc", "mIou")):
        super(EvalCallBack, self).__init__()
        self.network = network
        self.dataloader = dataloader
        self.eval_start_epoch = eval_start_epoch
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, best_ckpt_name)
        self.metrics_name = metrics_name

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            res = self.network.eval(self.dataloader, dataset_sink_mode=True)['SegmentationMetric']
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],\
                  ":INFO: epoch: {}, {}: {}, {}: {}".format(cur_epoch, self.metrics_name[0], \
                  res[0]*100, self.metrics_name[1], res[1]*100), flush=True)
            if res[1] >= self.best_res:
                self.best_res = res[1]
                self.best_epoch = cur_epoch
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],\
                   ":INFO: update best result: {}".format(res[1]*100), flush=True)
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        self.remove_ckpoint_file(self.best_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)
                    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],\
                     ":INFO: update best checkpoint at: {}".format(self.best_ckpt_path), flush=True)

    def end(self, run_context):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],\
        ":INFO: End training, the best {0} is: {1}, it's epoch is {2}".format(self.metrics_name[1],\
                        self.best_res*100, self.best_epoch), flush=True)
