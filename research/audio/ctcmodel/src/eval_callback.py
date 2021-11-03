# Copyright 2021 Huawei Technologies Co., Ltd
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
import stat
from mindspore import log as logger
from mindspore.train.callback import Callback
from mindspore import save_checkpoint


class EvalCallBack(Callback):
    """
       Evaluation callback when training.
    """

    def __init__(self, model, eval_ds, interval=1, save_best_ckpt=True,
                 ckpt_directory="./save", best_ckpt_name="best.ckpt", metrics_name="ler"):
        super(EvalCallBack, self).__init__()
        self.model = model
        self.eval_ds = eval_ds
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 10000000
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, best_ckpt_name)
        self.metrics_name = metrics_name

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def epoch_end(self, run_context):
        '''eval after intervals'''
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= 1 and cur_epoch % self.interval == 0:
            result = self.model.eval(self.eval_ds, dataset_sink_mode=False)
            res = result["ler"]
            print("epoch: {}, {}: {}".format(cur_epoch, self.metrics_name, res), flush=True)
            if res <= self.best_res:
                self.best_res = res
                self.best_epoch = cur_epoch
                print("update best result: {}".format(res), flush=True)
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        self.remove_ckpoint_file(self.best_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)
                    print("update best checkpoint at: {}".format(self.best_ckpt_path), flush=True)

    def end(self, run_context):
        '''eval end'''
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.metrics_name,
                                                                                     self.best_res,
                                                                                     self.best_epoch), flush=True)
