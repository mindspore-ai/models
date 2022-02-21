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
"""
utils.py
"""
import os
import stat
from datetime import datetime

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import save_checkpoint
from mindspore import log as logger
from mindspore.train.callback import Callback

class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class COMMON_AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', tb_writer=None):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.tb_writer = tb_writer
        self.cur_step = 1
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(self.name, self.val, self.cur_step)
        self.cur_step += 1

    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class TempLoss(nn.Cell):
    """A temp loss cell."""
    def construct(self, *inputs, **kwargs):
        return 0.1

class AccuracyMetric(nn.Metric):
    """R2plus1D Metric."""
    def __init__(self, dataset_size):
        super(AccuracyMetric, self).__init__()
        self.acc = 0.
        self.clear()
        self.softmax = nn.Softmax(axis=1)
        self.argmax = ops.ArgMaxWithValue(axis=1)
        self.reducesum = ops.ReduceSum()
        self.cast = ops.Cast()
        self.dataset_size = dataset_size
    def clear(self):
        """Resets the internal evaluation result to initial state."""
        self.acc = 0.

    def update(self, outputs, label):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
        probs = self.softmax(outputs)
        preds = self.argmax(probs)[0]
        self.acc += self.reducesum(self.cast(preds == label, mindspore.float32))

    def eval(self):
        return self.acc/self.dataset_size

class EvalCallBack(Callback):
    """
    Evaluation callback when training.

    Args:
        eval_function (function): evaluation function.
        eval_param_dict (dict): evaluation parameters' configure dict.
        interval (int): run evaluation interval, default is 1.
        eval_start_epoch (int): evaluation start epoch, default is 1.
        save_best_ckpt (bool): Whether to save best checkpoint, default is True.
        besk_ckpt_name (str): bast checkpoint name, default is `best.ckpt`.
        metrics_name (str): evaluation metrics name, default is `acc`.

    Returns:
        None

    Examples:
        >>> EvalCallBack(eval_function, eval_param_dict)
    """

    def __init__(self, network, dataloader, interval=1, eval_start_epoch=1, \
        save_best_ckpt=True, ckpt_directory="./", besk_ckpt_name="best.ckpt", f_model=None):
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
        self.bast_ckpt_path = os.path.join(ckpt_directory, besk_ckpt_name)
        self.f_model = f_model
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
            acc = self.network.eval(self.dataloader, dataset_sink_mode=True)['AccuracyMetric']

            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],\
                  ":INFO: epoch: {}, {}: {}".format(cur_epoch, "accuracy", acc*100), flush=True)

            if acc >= self.best_res:
                self.best_res = acc
                self.best_epoch = cur_epoch
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],\
                   ":INFO: update best result: {}".format(acc*100), flush=True)
                if self.save_best_ckpt:
                    if os.path.exists(self.bast_ckpt_path):
                        self.remove_ckpoint_file(self.bast_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.bast_ckpt_path)

                    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],\
                     ":INFO: update best checkpoint at: {}".format(self.bast_ckpt_path), flush=True)

    def end(self, run_context):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],\
        ":INFO: End training, the best {0} is: {1}, it's epoch is {2}".format("accuracy",\
                        self.best_res*100, self.best_epoch), flush=True)
