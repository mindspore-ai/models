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
"""Evaluation callback when training"""

import os
import stat
import time
import numpy as np
import mindspore as ms
from mindspore import save_checkpoint
from mindspore.train.callback import Callback


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, epoch_size, logger, lr, per_print_time=1, global_steps=0):
        super(LossCallBack, self).__init__()
        self.epoch_size = epoch_size
        self.logger = logger
        self.lr = lr
        self.global_steps = global_steps
        self.per_print_time = per_print_time
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        data_sink_mode = cb_params.get('dataset_sink_mode', True)
        if not data_sink_mode:
            if isinstance(loss, (tuple, list)):
                if isinstance(loss[0], ms.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                    loss = loss[0]

            if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
                loss = np.mean(loss.asnumpy())

            cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
            cur_epoch_num = cb_params.cur_epoch_num
            if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
                raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                    cb_params.cur_epoch_num, cur_step_in_epoch))

            if cur_step_in_epoch % self.per_print_time == 0:
                # pylint: disable=line-too-long
                per_step_time = 1000 * (time.time() - self.step_start_time) / self.per_print_time
                log_info = "epoch: [%s/%s] step: [%s/%s], lr: %.6f, loss: %.6f, per step time: %.3f ms" % (
                    cur_epoch_num, self.epoch_size, cur_step_in_epoch, cb_params.batch_num, self.lr[self.global_steps],
                    loss, per_step_time)
                self.logger.info(log_info)
                self.step_start_time = time.time()
        self.global_steps += 1

    def on_train_epoch_begin(self, run_context):
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        cur_epoch_num = cb_params.cur_epoch_num
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], ms.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        epoch_time = time.time() - self.epoch_start_time
        log_info = 'epoch: [%s/%s] loss: %.6f, epoch time: %.3f s, per step time: %.3f ms' % (
            cur_epoch_num, self.epoch_size, loss, epoch_time, epoch_time * 1000 / cb_params.batch_num)
        self.logger.info(log_info)


class ResumeCallback(Callback):
    def __init__(self, start_epoch=0):
        super(ResumeCallback, self).__init__()
        self.start_epoch = start_epoch

    def on_train_epoch_begin(self, run_context):
        run_context.original_args().cur_epoch_num += self.start_epoch


class EvalCallBack(Callback):
    """
    Evaluation callback when training.

    Args:
        eval_function (function): evaluation function.
        eval_param_dict (dict): evaluation parameters' configure dict.
        interval (int): run evaluation interval, default is 1.
        eval_start_epoch (int): evaluation start epoch, default is 1.
        save_best_ckpt (bool): Whether to save best checkpoint, default is True.
        best_ckpt_name (str): best checkpoint name, default is `best.ckpt`.
        metrics_name (str): evaluation metrics name, default is `acc`.

    Returns:
        None

    Examples:
        >>> EvalCallBack(eval_function, eval_param_dict)
    """

    def __init__(self, eval_function, eval_param_dict, interval=1, eval_start_epoch=1, rank_id=0, save_best_ckpt=True,
                 ckpt_directory="./", best_ckpt_name="best.ckpt", metrics_name="acc", logger=None):
        super(EvalCallBack, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        self.logger = logger
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0
        self.rank_id = rank_id
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
            self.logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            self.logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def on_train_epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            eval_start = time.time()
            res = self.eval_function(self.eval_param_dict)
            eval_cost = time.time() - eval_start
            self.logger.info("epoch: {}, {}: {}, eval_cost:{:.2f}".format(cur_epoch, self.metrics_name, res, eval_cost))
            if res >= self.best_res:
                if ms.context.get_context("enable_ge"):
                    from mindspore.train.callback import _set_cur_net
                    _set_cur_net(cb_params.train_network)
                    cb_params.train_network.exec_checkpoint_graph()
                self.best_res = res
                self.best_epoch = cur_epoch
                self.logger.info("update best result: %s", res)
                if self.save_best_ckpt and self.rank_id == 0:
                    if os.path.exists(self.best_ckpt_path):
                        self.remove_ckpoint_file(self.best_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)
                    self.logger.info("update best checkpoint at: %s", self.best_ckpt_path)

    def on_train_end(self, run_context):
        self.logger.info("End training, the best %s is: %s, the best %s epoch is %s" % (
            self.metrics_name, self.best_res, self.metrics_name, self.best_epoch))
