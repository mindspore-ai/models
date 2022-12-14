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

# This file was copied from project [mindspore][mindspore]

import os
import time

import numpy as np
from mindspore import save_checkpoint
from mindspore.common.tensor import Tensor
from mindspore.train.callback import Callback

from src.utils.local_adapter import get_rank_id

class CallbackSaveByIoU(Callback):
    """SaveCallback"""
    def __init__(self, eval_model, ds_eval, eval_period=1, eval_start=1, save_path=None):
        """init"""
        super(CallbackSaveByIoU, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.ins_mIoU = 0.
        self.cls_mIoU = 0.
        self.eval_period = eval_period
        self.save_path = save_path
        self.eval_start = eval_start

    def epoch_end(self, run_context):
        """epoch end"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        rank_id = get_rank_id()
        if ((cur_epoch + 1) % self.eval_period) == 0:
            if cur_epoch < self.eval_start:
                return
            if rank_id == 0:
                print("Start evaluate...")
            result = self.model.eval(self.ds_eval)
            cls_mIoU = result['IoU'][0]
            ins_mIoU = result['IoU'][1]
            if cls_mIoU > self.cls_mIoU:
                self.cls_mIoU = cls_mIoU
            if ins_mIoU > self.ins_mIoU:
                self.ins_mIoU = ins_mIoU
                file_name = f"best_model_dev_{rank_id}.ckpt"
                save_path = os.path.join(self.save_path, file_name)
                print("Save model...")
                save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=save_path)
            print(f"Device:{rank_id}, Epoce:{cur_epoch}, Instance mIoU:{ins_mIoU:.5f}, Class mIoU:{cls_mIoU:.5f}")

    def end(self, run_context):
        _ = run_context.original_args()
        rank_id = get_rank_id()
        print(f"Device:{rank_id}, Best Instance mIoU:{(self.ins_mIoU*100):.2f}%, Class mIoU:{(self.cls_mIoU*100):.2f}%")

class CallbackSaveByAcc(Callback):
    """SaveCallback"""
    def __init__(self, eval_model, ds_eval, eval_period=1, eval_start=1, save_path=None):
        """init"""
        super(CallbackSaveByAcc, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.acc = 0.
        self.eval_period = eval_period
        self.eval_start = eval_start
        self.save_path = save_path

    def epoch_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        rank_id = get_rank_id()
        if ((cur_epoch + 1) % self.eval_period) == 0:
            if cur_epoch < self.eval_start:
                return
            if rank_id == 0:
                print("Start evaluate...")
            result = self.model.eval(self.ds_eval)
            accuracy = result['acc']
            loss = result['loss']
            if accuracy > self.acc:
                self.acc = accuracy
                file_name = f"best_model_{rank_id}.ckpt"
                save_path = os.path.join(self.save_path, file_name)
                print("Save model...")
                save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=save_path)
            print(f"Device:{rank_id}, Epoce:{cur_epoch}, Accuracy:{accuracy:.5f}, Val loss:{loss:.5f}")

    def end(self, run_context):
        _ = run_context.original_args()
        rank_id = get_rank_id()
        print(f"On device {rank_id}, Best accuracy is {(self.acc*100):.2f}%")

class CheckLoss(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Raises:
        ValueError: If per_print_steps is not an integer or less than zero.
    """

    def begin(self, run_context):
        self.train_time = time.time()
        if get_rank_id() == 0:
            print('Train start at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        callback_params = run_context.original_args()
        loss = callback_params.net_outputs
        num_epoch = callback_params.cur_epoch_num

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        step_in_epoch = (callback_params.cur_step_num - 1) % callback_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                callback_params.cur_epoch_num, step_in_epoch))

        rank_id = get_rank_id()

        print(f"Device:{rank_id}, Epoce:{num_epoch}, Step:{step_in_epoch}, Train loss:{loss:.3f}", flush=True)

    def end(self, run_context):
        training_time = int(time.time() - self.train_time)
        if get_rank_id() == 0:
            print('Train end at {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
            print(f"Training time {training_time//3600}h{(training_time%3600)//60}m{(training_time%3600)%60}s")


class StopAtSteps(Callback):
    def __init__(self, stop_step=3):
        super(StopAtSteps, self).__init__()
        self.stop_step = stop_step

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num

        if step_num == self.stop_step:
            print("epoch:", epoch_num, " step:", step_num)
            run_context.request_stop()
