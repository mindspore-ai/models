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
"""Evaluation callback when training"""

import os

from mindspore import save_checkpoint
from mindspore.train.callback import Callback

from src.model_utils.config import config


class EvalCallBack(Callback):
    """
    Evaluation callback when training.

    Args:
        eval_function (function): evaluation function.
        eval_param_dict (dict): evaluation parameters' configure dict.
        interval (int): run evaluation interval, default is 1.
        eval_start_epoch (int): evaluation start epoch, default is 1.
        save_best_ckpt (bool): Whether to save best checkpoint, default is True.
        best_ckpt_name (str): best checkpoint name, default is `best.ckpt`.
        metrics_name (tuple): evaluation metrics name, default is (`acc`,`CMC`).

    Examples:
        >>> EvalCallBack(eval_function, eval_param_dict)
    """

    def __init__(self, eval_function, eval_param_dict, interval=1, eval_start_epoch=1, save_best_ckpt=True,
                 ckpt_directory="./", best_ckpt_name="best.ckpt", metrics_name=("acc", "CMC"), cmc_topk=(1, 5, 10)):
        super().__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_m_ap = 0
        self.best_cmc_scores = None
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, best_ckpt_name)
        self.metrics_name = metrics_name
        self.cmc_topk = cmc_topk

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            m_ap, cmc_scores = self.eval_function(self.eval_param_dict)
            print(f'Mean AP: {m_ap:4.1%}', flush=True)
            print(f'CMC Scores{config.dataset_name:>12}', flush=True)
            for k in self.cmc_topk:
                print(f'  top-{k:<4}{cmc_scores[config.dataset_name][k - 1]:12.1%}', flush=True)
            if m_ap >= self.best_m_ap:
                self.best_m_ap = m_ap
                self.best_cmc_scores = cmc_scores
                self.best_epoch = cur_epoch
                print(f"update best m_ap: {m_ap}", flush=True)
                if self.save_best_ckpt:
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)
                    print(f"update best checkpoint at: {self.best_ckpt_path}", flush=True)

    def end(self, run_context):
        """ Callback when total end """
        print(f"End training, the best epoch is {self.best_epoch}", flush=True)
        print("Best result:", flush=True)
        print(f'Mean AP: {self.best_m_ap:4.1%}', flush=True)
        print(f'CMC Scores{config.dataset_name:>12}', flush=True)
        for k in self.cmc_topk:
            print(f'  top-{k:<4}{self.best_cmc_scores[config.dataset_name][k - 1]:12.1%}', flush=True)
