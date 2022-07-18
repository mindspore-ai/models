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

import time
import os
import numpy as np
from mindspore.train.callback import Callback
from mindspore import save_checkpoint
from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config
from src.metric.calc import bpc, ppl


def doEval(net, dataset, tgt_len, ext_len, mem_len, eval_tgt_len):
    """Separate eval for valid and test"""
    net.set_train(tgt_len, ext_len, mem_len, eval_tgt_len, False)
    total_len, total_loss = 0, 0.
    idx = 0
    for data, target in dataset.create_tuple_iterator():
        loss = net(data, target, idx)
        idx = 1
        seq_len = target.shape[0]
        total_loss += seq_len * loss
        total_len += seq_len
        if net.is_first_iteration:
            net.add_flags_recursive(is_first_iteration=False)

    test_loss = total_loss / total_len
    test_loss = np.mean(test_loss.asnumpy())
    net.set_train(tgt_len, ext_len, mem_len, eval_tgt_len, True)
    if config.device_target == 'Ascend':
        test_loss -= config.pos_loss
    return test_loss


class EvalDuringTrain(Callback):
    def __init__(self, dataset, per_print_times, tgt_len, ext_len, mem_len,
                 eval_tgt_len):
        super(EvalDuringTrain, self).__init__()
        self.dataset = dataset
        self._per_print_times = per_print_times
        self.best_val_loss = None
        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.mem_len = mem_len
        self.eval_tgt_len = eval_tgt_len

    def step_end(self, run_context):
        """Called after each step finished."""
        device_id = get_device_id()
        cb_params = run_context.original_args()
        train_step = cb_params.cur_epoch_num
        if self._per_print_times != 0 and train_step % self._per_print_times == 0:
            eval_start_time = time.time()
            net = cb_params.network

            valid_loss = doEval(net, self.dataset, tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                                eval_tgt_len=self.eval_tgt_len)

            print('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(train_step // self._per_print_times, train_step,
                                                    (time.time() - eval_start_time), valid_loss)
            if config.dataset in ['enwik8', 'text8']:
                log_str += ' | valid bpc {:9.5f}'.format(bpc(valid_loss))
            else:
                log_str += ' | valid ppl {:9.3f}'.format(ppl(valid_loss))
            print(log_str)
            print('-' * 100)

            if not self.best_val_loss or valid_loss < self.best_val_loss:
                model_filename = os.path.join(config.train_url, 'model' + str(device_id) + '.ckpt')
                optimizer_filename = os.path.join(config.train_url, 'optimizer' + str(device_id) + '.ckpt')
                save_checkpoint(net, model_filename)
                save_checkpoint(cb_params.optimizer, optimizer_filename)
                self.best_val_loss = valid_loss
