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

import math
import time
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.train.callback import LossMonitor

from src.metric.calc import bpc, ppl
from src.model_utils.config import config


class TrainLogger(LossMonitor):
    def __init__(self, per_print_times, n_batch):
        super(TrainLogger, self).__init__(per_print_times)
        self.log_start_time = 0
        self.n_batch = n_batch
        self.train_loss = 0.0
        self.log_start_time = time.time()

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()
        train_step = cb_params.cur_epoch_num

        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        self.train_loss += loss
        if self._per_print_times != 0 and train_step % self._per_print_times == 0:
            epoch = math.ceil(train_step / self.n_batch)
            cur_loss = self.train_loss / self._per_print_times
            elapsed = time.time() - self.log_start_time
            batch = train_step % (self.n_batch + 1) + (0 if epoch == 1 else 1)
            optimizer = cb_params.optimizer
            train_step_t = Tensor(train_step, ms.int32)
            lr = optimizer.learning_rate(train_step_t).asnumpy()
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/step {:5.2f} | loss {:5.2f}'.format(epoch, train_step, batch, lr,
                                                                elapsed * 1000 / self._per_print_times, cur_loss)
            if config.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(bpc(cur_loss))
            else:
                log_str += ' | ppl {:9.3f}'.format(ppl(cur_loss))
            print(log_str)
            self.train_loss = 0.0
            self.log_start_time = time.time()
