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
"""
callback
"""
import os
from mindspore.train.callback import Callback
from mindspore import save_checkpoint


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

    def epoch_end(self, run_context):
        """Evaluate the network and save the best checkpoint (max top_1_accuracy)"""
        cb_params = run_context.original_args()
        result = self.model.eval(self.ds_eval)
        print("result:", result)
        if result['top_1_accuracy'] > self.top1:
            self.top1 = result['top_1_accuracy']
            file_name = 'attention_cluster_{}_{}.ckpt'.format(self.config.fc, self.config.natt)
            save_checkpoint(save_obj=cb_params.train_network,
                            ckpt_file_name=os.path.join(self.config.result_dir, file_name))
            print("Best checkpoint saved!")
