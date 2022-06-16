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
"""CallBack of jasper"""

import os
import logging
from mindspore import save_checkpoint
from mindspore.train.callback import Callback


class SaveCallback(Callback):
    """
    EvalCallback body
    """

    def __init__(self, path):

        super(SaveCallback, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.init_logger()
        self.interval = 10
        self.store_start_epoch = 10
        self.path = path

    def epoch_end(self, run_context):
        """
        select ckpt after some epoch
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if cur_epoch >= self.store_start_epoch and (cur_epoch - self.store_start_epoch) % self.interval == 0:
            message = '------------Epoch {} :start eval------------'.format(
                cur_epoch)
            self.logger.info(message)
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            filename = os.path.join(
                self.path, 'jasper' + '_' + str(cur_epoch) + '.ckpt')
            save_checkpoint(save_obj=cb_params.train_network,
                            ckpt_file_name=filename)
            message = '------------Epoch {} :training ckpt saved------------'.format(
                cur_epoch)
            self.logger.info(message)

    def init_logger(self):
        self.logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler('eval_callback.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
