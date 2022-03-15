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
"""callbacks"""
import datetime
import time

from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import RunContext
from mindspore.train.callback import _CallbackManager
from mindspore.train.callback import _InternalCallbackParam


class LossTimeMonitor(Callback):
    """loss time monitor"""
    def __init__(self, cfg):
        super().__init__()
        self.log_step_time = time.time()
        self.per_print_times = cfg.log_frequency_step
        self.steps_per_epoch = cfg.steps_per_epoch
        self.log_step_size = cfg.batch_size * cfg.device_num * self.per_print_times

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        cb_params.cur_step_num += 1
        if cb_params.cur_step_num % self.steps_per_epoch == 0:
            cb_params.cur_epoch_num += 1
        if cb_params.cur_step_num % self.per_print_times == 0:
            epoch = cb_params.cur_epoch_num
            cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
            loss = round(cb_params.net_outputs, 3)

            time_used = time.time() - self.log_step_time
            fps = round(self.log_step_size / time_used, 2)

            date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print(f"{date_time} epoch: {epoch}, iter: {cur_step_in_epoch}, "
                  f"loss: {loss}, fps: {fps} imgs/sec", flush=True)

            self.log_step_time = time.time()


class DetrCallbackManager:
    """detr callback manager"""
    def __init__(self, cfg, model):
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=cfg.steps_per_epoch,
            keep_checkpoint_max=cfg.keep_checkpoint_max
        )
        ckpt_cb = ModelCheckpoint(
            config=ckpt_config,
            directory=cfg.save_path,
            prefix='detr'
        )
        ltm_cb = LossTimeMonitor(cfg)

        cb_params = _InternalCallbackParam()
        cb_params.train_network = model
        cb_params.epoch_num = cfg.epochs
        cb_params.cur_epoch_num = 0
        cb_params.cur_step_num = 0
        cb_params.batch_num = cfg.steps_per_epoch * cfg.epochs
        self.cb_params = cb_params
        self.run_context = RunContext(self.cb_params)
        self.cb_manager = _CallbackManager([ckpt_cb, ltm_cb])
        self.cb_manager.begin(self.run_context)

    def __call__(self, loss):
        self.cb_params.net_outputs = loss
        self.cb_manager.step_end(self.run_context)


def get_callbacks(cfg, model):
    """get callbacks"""
    cb_manager = DetrCallbackManager(cfg, model)
    return cb_manager
