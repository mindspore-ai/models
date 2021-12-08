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
# ===========================================================================
""" Callback"""
import time
import numpy as np
from mindspore import Tensor
from mindspore.train.callback import Callback


def add_write(file_path, out_str):
    """ Write info"""
    with open(file_path, 'a+', encoding='utf-8') as file_out:
        file_out.write(out_str + '\n')


class EvalCallBack(Callback):
    """ Evaluate CallBack"""

    def __init__(self, model, eval_dataset, eval_per_steps, eval_file_path):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_file_path = eval_file_path
        self.eval_per_steps = eval_per_steps
        self.best_result = [0, 0, 0, 0]
        self.best_step = 0

    def epoch_end(self, run_context):
        """run after epoch end"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_global_step = cb_params.cur_step_num
        result = self.model.eval(self.eval_dataset)
        epoch_str = "Epoch: " + str(cur_epoch)
        print(epoch_str)
        print(result)
        result = result['Accuracy']
        if result[1] + result[2] > self.best_result[1] + self.best_result[2]:
            self.best_result = result
            self.best_step = cur_global_step
        print("Best Result: ", self.best_step, self.best_result)
        with open(self.eval_file_path, 'a+') as out_file:
            out_file.write(epoch_str + '\n')
            for p_at in result:
                out_file.write(str(p_at) + '\n')

    def step_end(self, run_context):
        """run after step end"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_global_step = cb_params.cur_step_num
        cur_step_in_epoch = (cur_global_step - 1) % cb_params.batch_num + 1
        if cur_global_step % self.eval_per_steps == 0:
            result = self.model.eval(self.eval_dataset)
            step_str = "Epoch:{}, step:{}, global_step:{}".format(str(cur_epoch), str(cur_step_in_epoch),
                                                                  str(cur_global_step))
            print(step_str)
            print(result)
            result = result['Accuracy']
            if result[1] + result[2] > self.best_result[1] + self.best_result[2]:
                self.best_result = result
                self.best_step = cur_global_step
            print("Best Result: ", self.best_step, self.best_result)
            with open(self.eval_file_path, 'a+') as out_file:
                out_file.write(step_str + '\n')
                for p_at in result:
                    out_file.write(str(p_at) + '\n')


class LossCallback(Callback):
    """ LossCallBack"""
    def __init__(self, loss_file_path, per_print_times=1):
        super(LossCallback, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.loss_file_path = loss_file_path

    def step_end(self, run_context):
        """ run after step_end """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            with open(self.loss_file_path, "a+") as loss_file:
                loss_file.write("{}\t{}\n".format(cb_params.cur_step_num, loss))
            print("epoch: %s step: %s global_step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch,
                                                                      cb_params.cur_step_num, loss), flush=True)


class TimeMonitor(Callback):
    """ TimeMonitor"""
    def __init__(self, data_size):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = None
        self.step_time = None

    def epoch_begin(self, run_context):
        """ run after epoch_begin """
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """ run after epoch_end """
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / self.data_size
        print("epoch time: {0}, per step time: {1}".format(epoch_mseconds, per_step_mseconds), flush=True)

    def step_begin(self, run_context):
        """ run after step_begin """
        self.step_time = time.time()

    def step_end(self, run_context):
        """ run after step_end """
        step_mseconds = (time.time() - self.step_time) * 1000
        print(f"step time {step_mseconds}", flush=True)
