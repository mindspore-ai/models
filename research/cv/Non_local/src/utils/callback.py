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
"""
callback
"""
import os
from mindspore.train.callback import Callback
from mindspore import save_checkpoint
from mindspore.communication.management import get_rank


class SaveCallback(Callback):
    """
    Save the best checkpoint (max top_1_accuracy) during training
    """

    def __init__(self, eval_model, ds_eval, config):
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.top1 = 0
        self.config = config
        self.epoch = 0

    def epoch_end(self, run_context):
        """Evaluate the network and save the best checkpoint (max top_1_accuracy)"""
        cb_params = run_context.original_args()
        self.epoch += 1
        if self.epoch > 90 or self.epoch % 5 == 0:
            result = self.model.eval(self.ds_eval, dataset_sink_mode=False)
            print("result:", result)
            if result['top_1_accuracy'] > self.top1:
                self.top1 = result['top_1_accuracy']
                file_name = 'nl_{}.ckpt'.format(self.config.dataset)
                if self.config.distributed and get_rank() == 0 or not self.config.distributed:
                    save_checkpoint(save_obj=cb_params.train_network,
                                    ckpt_file_name=os.path.join(self.config.result_path, file_name))
                    print("Best checkpoint saved!")
