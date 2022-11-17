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
"""callback function"""

from mindspore.train.callback import Callback

from src.args import args


class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, src_url, train_url, save_freq=50):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.src_url = src_url
        self.train_url = train_url
        self.save_freq = save_freq
        self.best_acc = 0.

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        result = self.model.eval(self.eval_dataset)
        if result["acc"] > self.best_acc:
            self.best_acc = result["acc"]
        print("epoch: %s acc: %s, best acc is %s" %
              (cb_params.cur_epoch_num, result["acc"], self.best_acc), flush=True)
        if args.run_modelarts:
            import moxing as mox
            if cur_epoch_num % self.save_freq == 0:
                mox.file.copy_parallel(src_url=self.src_url, dst_url=self.train_url)
