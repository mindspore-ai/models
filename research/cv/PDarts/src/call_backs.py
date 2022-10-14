# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train callbacks"""
import os
import time

import numpy as np

from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback._callback import Callback


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
        """
        Call at each step end.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                "step: {}. Invalid loss, terminating training.".format(cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("step: %s, loss is %s" %
                  (cur_step_in_epoch, loss), flush=True)


class Val_Callback(Callback):
    """
    Valid the test data at every epoch.
    """

    def __init__(self, model, train_dataset, val_dataset, checkpoint_path, prefix,
                 network, img_size, rank_id=0, is_eval_train_dataset='False'):
        super(Val_Callback, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_path = checkpoint_path
        self.max_val_acc = 0
        self.prefix = prefix
        self.network = network
        self.img_size = img_size
        self.rank_id = rank_id
        self.is_eval_train_dataset = is_eval_train_dataset

    def epoch_end(self, run_context):
        """
        Call at each epoch end.
        """
        if self.is_eval_train_dataset == 'True':
            start = time.time()
            train_result = self.model.eval(
                self.train_dataset, dataset_sink_mode=False)
            end = time.time()
            print("==========train metrics:" + str(train_result) + " use times:" +
                  str((end - start) * 1000) + "ms=========================")
        start = time.time()
        val_result = self.model.eval(self.val_dataset, dataset_sink_mode=False)
        end = time.time()
        print("==========val metrics:" + str(val_result) + " use times:" +
              str((end - start) * 1000) + "ms=========================")
        val_acc = val_result['top_1_accuracy']
        if val_acc > self.max_val_acc:
            print('=================save checkpoint....====================')
            self.max_val_acc = val_acc
            cb_params = run_context.original_args()
            epoch = cb_params.cur_epoch_num
            model_info = self.prefix + '_id' + str(self.rank_id) + \
                '_epoch' + str(epoch) + '_valacc' + str(val_acc)
            if self.checkpoint_path.startswith('s3://') or self.checkpoint_path.startswith('obs://'):
                save_path = '/cache/save_model/'
            else:
                save_path = self.checkpoint_path
            save_path = os.path.join(save_path, model_info)
            # save checkpoint
            ckpt_path = os.path.join(save_path, 'checkpoint')
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            ckpt_file = os.path.join(ckpt_path, 'model_checkpoint.ckpt')
            save_checkpoint(cb_params.train_network, ckpt_file)
            if self.checkpoint_path.startswith('s3://') or self.checkpoint_path.startswith('obs://'):
                from moxing.framework import file
                file.copy_parallel(save_path, os.path.join(
                    self.checkpoint_path, model_info))
            print('==============save checkpoint finished===================')
        print(f'The best accuracy is {self.max_val_acc}')


class Set_Attr_CallBack(Callback):
    """
    Set drop_path_prob and epoch_mask.

    """

    def __init__(self, model, drop_path_prob, epochs, layers, batch_size):
        super(Set_Attr_CallBack, self).__init__()
        self.model = model
        self.drop_path_prob = drop_path_prob
        self.epochs = epochs
        self.layers = layers
        self.batch_size = batch_size

    def epoch_begin(self, run_context):
        """
        Call at each epoch begin.
        """
        self.model.drop_path_prob = self.drop_path_prob * 300 / self.epochs
        keep_prob = 1. - self.model.drop_path_prob
        self.model.epoch_mask = []
        for _ in range(self.layers):
            layer_mask = []
            for _ in range(5 * 2):
                mask = np.array([np.random.binomial(1, p=keep_prob)
                                 for k in range(self.batch_size)])
                mask = mask[:, np.newaxis, np.newaxis, np.newaxis]
                mask = Tensor(mask, mstype.float16)
                layer_mask.append(mask)
            self.model.epoch_mask.append(layer_mask)
