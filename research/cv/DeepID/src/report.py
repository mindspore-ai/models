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
"""Reporter class."""
import logging
import time
import datetime
from mindspore import Tensor


class Reporter(logging.Logger):
    """
    This class includes several functions that can save images/checkpoints and print/save logging information.

    Args:
        args (class): Option class.
    """

    def __init__(self, num_iters):
        super(Reporter, self).__init__("DeepID")

        self.epoch = 0
        self.step = 0
        self.print_iter = 50
        self.deepid_loss = []
        self.total_step = num_iters
        self.runs_step = 0

    def epoch_start(self):
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()
        self.step = 0
        self.epoch += 1
        self.deepid_loss = []


    def print_info(self, start_time, step, lossG):
        """print log after some steps."""
        resID, _ = self.return_loss_array(lossG)
        if self.step % self.print_iter == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            losses = "Loss: [{:.2f}], Acc: [{:.2f}%].".format(resID[0], resID[1]*100)
            print("Step [{}/{}] Elapsed [{} s], {}".format(
                step + 1, self.total_step, elapsed[:-7], losses))

    def return_loss_array(self, lossID):
        """Transform output to loooooss array"""
        resID = []
        deepid_list = ['deepid_loss', 'acc']
        dict_ID = {'deepid_loss': 0., 'acc': 0.}
        self.deepid_loss.append(float(lossID[0].asnumpy()))
        for i, item in enumerate(lossID):
            resID.append(float(item.asnumpy()))
            dict_ID[deepid_list[i]] = Tensor(float(item.asnumpy()))

        return resID, dict_ID

    def epoch_end(self):
        """print log and save cgeckpoints when epoch end."""
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        pre_step_time = epoch_cost / self.step
        mean_loss_deepID = sum(self.deepid_loss) / self.step

        self.info("Epoch [{}] total cost: {:.2f} ms, pre step: {:.2f} ms, loss: {:.2f}".format(
            self.epoch, epoch_cost, pre_step_time, mean_loss_deepID))
