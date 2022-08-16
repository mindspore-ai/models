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
"""saveCallBack"""
import time
from mindspore import load_param_into_net, load_checkpoint
from mindspore.train.callback import Callback
from src.utils.evaluate import test

class SaveCallback(Callback):
    """
    define savecallback, save best model while training.
    """
    def __init__(self, model, query_dataset, gallery_dataset,
                 vcolor2label, vtype2label, epoch_per_eval, max_epoch, path, step_size, device_id):
        super(SaveCallback, self).__init__()
        self.model = model
        self.query_dataset = query_dataset
        self.gallery_dataset = gallery_dataset
        self.vcolor2label = vcolor2label
        self.vtype2label = vtype2label
        self.epoch_per_eval = epoch_per_eval
        self.max_epoch = max_epoch
        self.path = path
        self.step_size = step_size
        self.device_id = device_id

    def epoch_end(self, run_context):
        """
        eval and save model while training.
        """
        t1 = time.time()
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        file_name = self.path + "MultipleNet-" + str(cur_epoch) + "_" + str(self.step_size) + ".ckpt"
        param_dict = load_checkpoint(file_name)
        load_param_into_net(self.model, param_dict)

        print("\n--------------------{} / {}--------------------\n".format(cur_epoch, self.max_epoch))

        print("----------device is {}".format(self.device_id))

        _ = test(self.model, True, True, self.query_dataset, self.gallery_dataset, \
            self.vcolor2label, self.vtype2label, return_distmat=True)
        t2 = time.time()
        print("Eval in training Time consume: ", t2 - t1, "\n")
