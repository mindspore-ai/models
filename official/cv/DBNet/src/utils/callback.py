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
"""Monitor the result of DBNet."""
import os
import time
import numpy as np

import mindspore as ms
from mindspore.train.callback import Callback

from src.datasets.load import create_dataset
from src.modules.model import get_dbnet
from .metric import AverageMeter
from .eval_utils import WithEval


class DBNetMonitor(Callback):
    """
    Monitor the result of DBNet.
    If the loss is NAN or INF, it will terminate training.
    Note:
        If per_print_times is 0, do not print loss.
    Args:
        config(class): configuration class.
        train_net(nn.Cell): Train network.
        per_print_times (int): How many steps to print once loss. During sink mode, it will print loss in the
                               nearest step. Default: 1.
    Raises:
        ValueError: If per_print_times is not an integer or less than zero.
    """

    def __init__(self, config, train_net, per_print_times=1):
        super(DBNetMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.config = config

        self.loss_avg = AverageMeter()
        self.rank_id = config.rank_id
        self.run_eval = config.run_eval
        self.eval_iter = config.eval_iter
        if self.run_eval:
            config.backbone.pretrained = False
            eval_net = get_dbnet(config.net, config, isTrain=False)
            self.eval_net = WithEval(eval_net, config)
            val_dataset, _ = create_dataset(config, False)
            self.val_dataset = val_dataset.create_dict_iterator(output_numpy=True)
            self.max_f = 0.0
        self.train_net = train_net
        self.epoch_start_time = time.time()
        self.checkout_path = config.output_dir
        if not os.path.isdir(self.checkout_path) and self.config.rank_id == 0:
            os.makedirs(self.checkout_path, exist_ok=True)
        else:
            print("WARNING: The checkpoint path is already exist. The saved checkpoint file may be overwritten.",
                  flush=True)

    def on_train_step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if cb_params.net_outputs is not None:
            if isinstance(loss, tuple):
                if loss[1]:
                    print("==========overflow!==========", flush=True)
                loss = loss[0]
            loss = loss.asnumpy()
        else:
            print("custom loss callback class loss is None.", flush=True)
            return

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if cur_step_in_epoch == 1:
            self.loss_avg = AverageMeter()
        self.loss_avg.update(loss)

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            loss_log = "[%s] rank%d epoch: %d step: %2d, loss is %.6f" % \
                       (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), self.config.rank_id,
                        cb_params.cur_epoch_num, cur_step_in_epoch, np.mean(self.loss_avg.avg))
            print(loss_log, flush=True)

    def on_train_epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, run_context):
        """
        Called after each training epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        epoch_time = (time.time() - self.epoch_start_time) * 1000
        time_log = "[%s] rank%d epoch: %d cast %2f ms, per tep time: %2f ms" % \
                   (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), self.config.rank_id,
                    cur_epoch, epoch_time, epoch_time / cb_params.batch_num)
        print(time_log, flush=True)
        train_net = self.train_net
        if self.run_eval and cur_epoch % self.eval_iter == 0:
            ms.save_checkpoint(train_net,
                               os.path.join(self.checkout_path, f"cur_epoch_rank{self.config.rank_id}.ckpt"))
            ms.load_checkpoint(os.path.join(self.checkout_path, f"cur_epoch_rank{self.config.rank_id}.ckpt"),
                               self.eval_net.model)

            self.eval_net.model.set_train(False)
            metrics, fps = self.eval_net.eval(self.val_dataset, show_imgs=self.config.eval.show_images)

            cur_f = metrics['fmeasure'].avg
            print(f"\nrank{self.config.rank_id} current epoch is {cur_epoch}\n"
                  f"FPS: {fps}\n"
                  f"Recall: {metrics['recall'].avg}\n"
                  f"Precision: {metrics['precision'].avg}\n"
                  f"Fmeasure: {metrics['fmeasure'].avg}\n", flush=True)
            if cur_f >= self.max_f:
                print(f"update best ckpt at epoch {cur_epoch}, best fmeasure is {cur_f}\n", flush=True)
                ms.save_checkpoint(self.eval_net.model,
                                   os.path.join(self.checkout_path, f"best_rank{self.config.rank_id}.ckpt"))
                self.max_f = cur_f

    def on_train_end(self, run_context):
        print(f"rank{self.config.rank_id} best fmeasure is {self.max_f}", flush=True)
