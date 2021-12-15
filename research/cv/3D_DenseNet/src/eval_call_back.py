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
This module is used for show information during training
"""
from src.metrics import train_dice
import numpy as np
import mindspore
from mindspore.train.callback import Callback

class StepLossAccInfo(Callback):
    """
    custom callback function
    There are a total of 9 data sets used in the training,
    the DICE coefficient of the validation set is output at the 9th step
    """
    def __init__(self, model, eval_dataset, steps_loss, steps_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.steps_loss = steps_loss
        self.steps_eval = steps_eval

    def step_end(self, run_context):
        """
        The custom step_end function
        """
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))
        if cur_step % 9 == 0:
            for images_val, targets_val in self.eval_dataset:
                outputs_val = self.model(images_val)
                outputs_val = outputs_val.transpose(0, 2, 3, 4, 1)
                predicted = mindspore.ops.Argmax(-1)(outputs_val)
                #Compute dice
                predicted_val = predicted.asnumpy()
                targets_val = targets_val.asnumpy()
                dsc = []
                #ignore Background 0
                for i in range(1, 4):
                    dsc_i = train_dice(predicted_val, targets_val, i)
                    dsc.append(dsc_i)
                dsc = np.mean(dsc)
                global_dsc = dsc
            print("valid_dice:", global_dsc)
            self.steps_eval["step"].append(cur_step)
            self.steps_eval["dice"].append(global_dsc)
