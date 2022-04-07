# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Defined eval callback for mmoe.
"""
import os
import time

from sklearn.metrics import roc_auc_score

from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint

from src.model_utils.config import config


class EvalCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note
        If per_print_times is 0 do not print loss.
    """

    def __init__(self, net, ds_eval, ckpt_path, rank_id):
        super(EvalCallBack, self).__init__()
        self.net = net
        self.ds_eval = ds_eval
        self.ckpt_path = ckpt_path
        self.rank_id = rank_id
        self.max_income_auc = 0
        self.max_marital_auc = 0
        self.max_income_marital_auc_avg = 0

    def epoch_end(self, run_context):
        start_time = time.time()
        eval_dataloader = self.ds_eval.create_tuple_iterator()

        income_output_list = []
        marital_output_list = []

        income_label_list = []
        marital_label_list = []

        data_type = mstype.float16 if config.device_target == 'Ascend' else mstype.float32

        print('start infer...')
        self.net.set_train(False)
        for data, income_label, marital_label in eval_dataloader:
            output = self.net(Tensor(data, data_type))

            income_output_list.extend(output[0].asnumpy().flatten().tolist())
            marital_output_list.extend(output[1].asnumpy().flatten().tolist())

            income_label_list.extend(income_label.asnumpy().flatten().tolist())
            marital_label_list.extend(
                marital_label.asnumpy().flatten().tolist())

        if len(income_output_list) != len(income_label_list):
            raise RuntimeError(
                'income_output.size() is not equal income_label.size().')
        if len(marital_output_list) != len(marital_label_list):
            raise RuntimeError(
                'marital_output.size is not equal marital_label.size().')
        print('infer data finished, start eval...')
        income_auc = roc_auc_score(income_label_list, income_output_list)
        marital_auc = roc_auc_score(marital_label_list, marital_output_list)

        eval_time = int(time.time() - start_time)
        print(
            f"result : income_auc={income_auc}, marital_auc={marital_auc}, use time {eval_time}s")

        cb_params = run_context.original_args()

        if income_auc > self.max_income_auc:
            self.max_income_auc = income_auc
            ckpt_file_name = 'best_income_auc.ckpt'
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            save_path = os.path.join(self.ckpt_path, ckpt_file_name)
            save_checkpoint(cb_params.train_network, save_path)

        if marital_auc > self.max_marital_auc:
            self.max_marital_auc = marital_auc
            ckpt_file_name = 'best_marital_auc.ckpt'
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            save_path = os.path.join(self.ckpt_path, ckpt_file_name)
            save_checkpoint(cb_params.train_network, save_path)

        income_marital_auc_avg = (income_auc + marital_auc) / 2
        if income_marital_auc_avg > self.max_income_marital_auc_avg:
            self.max_income_marital_auc_avg = income_marital_auc_avg
            ckpt_file_name = 'best_income_marital_auc_avg.ckpt'
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            save_path = os.path.join(self.ckpt_path, ckpt_file_name)
            save_checkpoint(cb_params.train_network, save_path)

        print(
            f'The best income_auc is {self.max_income_auc}, \
            the best marital_auc is {self.max_marital_auc}, \
            the best income_marital_auc_avg is {self.max_income_marital_auc_avg}')
