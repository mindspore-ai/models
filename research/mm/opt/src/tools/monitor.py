# Copyright 2020 Huawei Technologies Co., Ltd
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
"""TimeMonitor Callback class."""
import os
from multiprocessing import Process
import moxing as mox
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.train.callback._callback import Callback
from data import task2id

id2task = {}
for k, v in task2id.items():
    id2task[v] = k


class LossMonitor(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        per_print_times (int): Print the loss each every time. Default: 1.

    Raises:
        ValueError: If print_step is not an integer or less than zero.
    """

    def __init__(self, per_print_times=1):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        print("==========", loss)
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                task_name = id2task[int(loss[3].asnumpy())]
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s, task_name: %s, step: %s, loss is %s" % (
                cb_params.cur_epoch_num, task_name, cur_step_in_epoch, loss), flush=True)


class LossMonitorSingleTask(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        per_print_times (int): Print the loss each every time. Default: 1.

    Raises:
        ValueError: If print_step is not an integer or less than zero.
    """

    def __init__(self, per_print_times=1):
        super(LossMonitorSingleTask, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        print("==========", loss)
        # if isinstance(loss, (tuple, list)):
        #     if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
        #         task_name = id2task[int(loss[3].asnumpy())]
        #         loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s, step: %s, loss is %s" %
                  (cb_params.cur_epoch_num, cur_step_in_epoch, loss), flush=True)



class UploadCheckpoint(Callback):
    """Upload Checkpoint"""
    def __init__(self, target_save_dir, upload_frequence=10, rank_id=0):
        self.target_save_dir = target_save_dir
        self.upload_frequence = upload_frequence
        self.rank_id = str(rank_id)

    def epoch_end(self, run_context):
        """Epoch end"""
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.upload_frequence == 0:
            print("Find ckpt dirs: ", os.listdir("/cache/ckpt"))
            target_path = os.path.join(
                self.target_save_dir + "rank_{}".format(self.rank_id))
            print('target dir is:', target_path)
            mox.file.copy_parallel(src_url="/cache/ckpt/" + "rank_{}".format(self.rank_id),
                                   dst_url=target_path)
            print("Upload ckpt succeed!")


class UploadLog(Callback):
    """Upload Log"""
    def __init__(self, target_save_dir, upload_frequence=10, rank_id=0):
        self.target_save_dir = target_save_dir
        self.upload_frequence = upload_frequence
        self.rank_id = rank_id

    def epoch_end(self, run_context):
        """epoch end"""
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.upload_frequence == 0 and self.rank_id % 8 == 0:
            print("Find log dirs: ", os.listdir("/tmp/log"))
            target_path = os.path.join(
                self.target_save_dir + "rank_{}_{}".format(str(self.rank_id), str(self.rank_id + 7)), "train.log")
            print('target dir is:', target_path)
            if mox.file.exists(target_path):
                mox.file.remove(target_path, recursive=True)
            process = Process(target=mox.file.copy_parallel, args=("/tmp/log/train.log", target_path),
                              name="file_sync1")
            process.start()
            print("Upload log succeed!")


class UploadSummary(Callback):
    """Upload Summary"""
    def __init__(self, target_save_dir, upload_frequency=50):
        self.target_save_dir = target_save_dir
        self.upload_frequence = upload_frequency

    def epoch_end(self, run_context):
        """epoch end"""
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.upload_frequence == 0:
            target_path = self.target_save_dir
            if mox.file.exists(target_path):
                mox.file.remove(target_path, recursive=True)

            process = Process(target=mox.file.copy_parallel, args=("/cache/summary_dir", target_path),
                              name="file_sync2")

            process.start()
            print("Upload summary succeed!")
