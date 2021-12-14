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

from mindspore.train.callback import Callback

class EvalCallBack(Callback):
    def __init__(self, model, eval_dataset, eval_per_epoch, eval_log):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.eval_log = eval_log
        self.best_epoch = None

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            print("\nValidating...")
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.eval_log["Epoch"].append(cur_epoch)
            self.eval_log["Val_Loss"].append(acc["Loss"])
            self.eval_log["Val_Top1-Acc"].append(acc["Top1-Acc"])
            self.eval_log["Val_Top5-Acc"].append(acc["Top5-Acc"])
            print("\tValidation performance:")
            print(f"\tEpoch: {cur_epoch}, Val Loss: {acc['Loss']},",
                  f"Val Top1-Acc: {acc['Top1-Acc']}, Val Top5-Acc: {acc['Top5-Acc']}")
            if (self.best_epoch is None) or \
               (acc["Loss"] < self.eval_log["Val_Loss"][(self.best_epoch // self.eval_per_epoch) - 1]):
                self.best_epoch = cur_epoch
                print(f"\tValidation performance improved! New best epoch: {self.best_epoch}\n")
            else:
                print(f"\tValidation performance did not improved. Current best epoch: {self.best_epoch}\n")
