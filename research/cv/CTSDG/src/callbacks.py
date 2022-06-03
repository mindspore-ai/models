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
import os
import time
from pathlib import Path

from mindspore import save_checkpoint
from mindspore.train.callback import Callback
from mindspore.train.callback import RunContext
from mindspore.train.callback import _CallbackManager
from mindspore.train.callback import _InternalCallbackParam

_S_IWRITE = 128


class LossTimeMonitor(Callback):
    """loss time monitor"""
    def __init__(self, cfg):
        super().__init__()
        self.log_step_time = time.time()
        self.per_print_times = cfg.log_frequency_step

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        cb_params.cur_step_num += 1
        if cb_params.cur_step_num % self.per_print_times == 0:
            loss_g, loss_d = cb_params.net_outputs
            loss_g = round(loss_g, 3)
            loss_d = round(loss_d, 3)

            time_used = time.time() - self.log_step_time
            per_step_time = round(1e3 * time_used / self.per_print_times, 2)

            date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print(f"{date_time} iter: {cb_params.cur_step_num}, loss_g: {loss_g}, "
                  f"loss_d: {loss_d}, step time: {per_step_time} ms", flush=True)

            self.log_step_time = time.time()


class CTSDGModelCheckpoint(Callback):
    """CTSDG ModelCheckpoint"""
    def __init__(self, cfg):
        super().__init__()
        self.save_checkpoint_steps = cfg.save_checkpoint_steps
        self.keep_checkpoint_max = cfg.keep_checkpoint_max
        self.checkpoint_filelist = []
        self.save_path = cfg.save_path
        self.total_steps = cfg.total_steps
        Path(self.save_path).mkdir(exist_ok=True, parents=True)

    def _remove_ckpoint_file(self, file_name, is_g):
        """Remove the specified checkpoint file from this checkpoint manager
        and also from the directory."""
        try:
            os.chmod(file_name, _S_IWRITE)
            os.remove(file_name)
            if is_g:
                self.checkpoint_filelist.remove(file_name)
        except OSError:
            print(f"OSError, failed to remove the older ckpt file {file_name}.", flush=True)
        except ValueError:
            print(f"ValueError, failed to remove the older ckpt file {file_name}.", flush=True)

    def _remove_oldest(self):
        """remove oldest checkpoint file"""
        ckpoint_files = sorted(self.checkpoint_filelist, key=os.path.getmtime)
        file_to_remove = Path(ckpoint_files[0])
        name_g = file_to_remove.name
        name_d = name_g.replace('generator_', 'discriminator_')
        file_name_g = file_to_remove.as_posix()
        file_name_d = (file_to_remove.parent / name_d).as_posix()
        self._remove_ckpoint_file(file_name_g, True)
        self._remove_ckpoint_file(file_name_d, False)

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step % self.save_checkpoint_steps == 0 or cur_step == self.total_steps:
            g_name = os.path.join(self.save_path, f'generator_{cur_step:06d}.ckpt')
            save_checkpoint(cb_params.train_network_g, g_name)
            d_name = os.path.join(self.save_path, f'discriminator_{cur_step:06d}.ckpt')
            save_checkpoint(cb_params.train_network_d, d_name)
            self.checkpoint_filelist.append(g_name)

        if len(self.checkpoint_filelist) > self.keep_checkpoint_max:
            self._remove_oldest()


class CTSDGCallbackManager:
    """ctsdg callback manager"""
    def __init__(self, cfg, model_g, model_d):
        ckpt_cb = CTSDGModelCheckpoint(cfg)
        ltm_cb = LossTimeMonitor(cfg)

        cb_params = _InternalCallbackParam()
        cb_params.train_network_g = model_g
        cb_params.train_network_d = model_d
        cb_params.cur_step_num = cfg.start_iter
        self.cb_params = cb_params
        self.run_context = RunContext(self.cb_params)
        self.cb_manager = _CallbackManager([ltm_cb, ckpt_cb])
        self.cb_manager.begin(self.run_context)

    def __call__(self, losses):
        self.cb_params.net_outputs = losses
        self.cb_manager.step_end(self.run_context)


def get_callbacks(cfg, model_g, model_d, finetune):
    """get callbacks"""
    cb_manager = CTSDGCallbackManager(cfg, model_g, model_d)
    print('==============================', flush=True)
    if finetune:
        print('Start finetune', flush=True)
    else:
        print(f'Start training', flush=True)
    return cb_manager
