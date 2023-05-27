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

import os
import shutil
import mindspore as ms
from mindspore.train.callback import Callback


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
        metrics_name (str): evaluation metrics name, default is `acc`.

    Returns:
        None

    Examples:
        >>> EvalCallBack(eval_function, eval_param_dict)
    """

    def __init__(self, config, net, apply_eval, datasetsize, mindrecord_path, anno_json, checkpoint_path):
        super(EvalCallBack, self).__init__()
        self.faster_rcnn_eval = apply_eval
        self.mindrecord_path = mindrecord_path
        self.anno_json = anno_json
        self.datasetsize = datasetsize
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.net = net
        self.best_epoch = 0
        self.best_res = 0
        self.best_ckpt_path = os.path.abspath("./best_ckpt")

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        ckpt_file_name = "faster_rcnn-{}_{}.ckpt".format(cur_epoch, self.datasetsize)
        checkoint_path = os.path.join(self.checkpoint_path, ckpt_file_name)
        self.config.current_epoch = cur_epoch
        res = 0
        if self.config.current_epoch % self.config.interval == 0 or self.config.current_epoch == self.config.epoch_size:
            res = self.faster_rcnn_eval(self.net, self.config, self.mindrecord_path, checkoint_path, self.anno_json)

        if res > self.best_res:
            self.best_epoch = cur_epoch
            self.best_res = res

            if os.path.exists(self.best_ckpt_path):
                shutil.rmtree(self.best_ckpt_path)

            os.mkdir(self.best_ckpt_path)
            ms.save_checkpoint(cb_params.train_network, os.path.join(self.best_ckpt_path, "best.ckpt"))

            print("update best result: {} in the {} th epoch".format(self.best_res, self.best_epoch), flush=True)

    def end(self, run_context):
        print("End training the best {0} epoch is {1}".format(self.best_res, self.best_epoch), flush=True)
