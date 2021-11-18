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
Custom callback and related RMSE metric
"""
import os
import numpy as np
from mindspore.dataset.core.validator_helpers import INT32_MAX
from mindspore.train.callback import Callback
from mindspore import save_checkpoint
from mindspore.nn import Metric


class RMSE(Metric):
    """
    RMSE metric for choosing the best checkpoint
    """

    def __init__(self, max_val):
        super(RMSE, self).__init__()
        self.clear()
        self.max_val = max_val

    def clear(self):
        """Clears the internal evaluation result"""
        self._squared_error_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        """Calculate and update internal result"""
        if len(inputs) != 2:
            raise ValueError('RMSE metric need 2 inputs (preds, targets), but got {}'.format(len(inputs)))
        preds = self._convert_data(inputs[0])
        targets = self._convert_data(inputs[1])
        targets = targets.reshape((-1, targets.shape[2]))
        squared_error_sum = np.power(targets - preds, 2)
        self._squared_error_sum += squared_error_sum.sum()
        self._samples_num += np.size(targets)

    def eval(self):
        """Calculate evaluation result at the end of each epoch"""
        if self._samples_num == 0:
            raise RuntimeError('The number of input samples must not be 0.')
        return np.sqrt(self._squared_error_sum / self._samples_num) * self.max_val


class SaveCallback(Callback):
    """
    Save the best checkpoint (minimum RMSE) during training
    """

    def __init__(self, eval_model, ds_eval, config):
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.rmse = INT32_MAX
        self.config = config

    def epoch_end(self, run_context):
        """Evaluate the network and save the best checkpoint (minimum RMSE)"""
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        cb_params = run_context.original_args()
        file_name = self.config.dataset + '_' + str(self.config.pre_len) + '.ckpt'
        if self.config.save_best:
            result = self.model.eval(self.ds_eval)
            print('Eval RMSE:', '{:.6f}'.format(result['RMSE']))
            if result['RMSE'] < self.rmse:
                self.rmse = result['RMSE']
                save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=os.path.join('checkpoints', file_name))
                print("Best checkpoint saved!")
        else:
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=os.path.join('checkpoints', file_name))
