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

"""Local adapter"""

import os
import time

from mindspore.train.callback import Callback


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.count = 0
        self.loss_x = 0
        self.loss_u = 0
        self.loss_c = 0
        self.scaling_sens = 0
        self.rank_id = rank_id
        self.time_stamp_first = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        self.count += 1
        self.loss_x += float(loss[1].asnumpy())
        self.loss_u += float(loss[2].asnumpy())
        self.loss_c += float(loss[3].asnumpy())

        if self.count >= 1:
            time_stamp_current = time.time()
            total_x = self.loss_x / self.count
            total_u = self.loss_u / self.count
            total_c = self.loss_c / self.count

            loss_file = open("./loss_{}.log".format(self.rank_id), "a+")
            loss_file.write("%lu epoch: %s step: %s total_x: %.5f, total_u: %.5f, total_c: %.5f" %
                            (time_stamp_current - self.time_stamp_first, cb_params.cur_epoch_num, cur_step_in_epoch,
                             total_x, total_u, total_c))
            loss_file.write("\n")
            loss_file.close()
            self.count = 0
            self.count = 0
            self.loss_x = 0
            self.loss_u = 0
            self.loss_c = 0


def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    return "Local Job"
