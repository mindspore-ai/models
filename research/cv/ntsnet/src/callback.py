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
import os

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops
from mindspore import save_checkpoint
from mindspore.train.callback import Callback


class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, save_path):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.best_acc = 0.
        self.save_path = os.path.join(save_path, "best.ckpt")
        self.print = ops.Print()
        os.makedirs(save_path, exist_ok=True)

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """

        cb_params = run_context.original_args()
        self.model.set_train(False)
        success_num = 0.0
        total_num = 0.0
        for _, (image, label) in enumerate(self.eval_dataset):
            image_data = Tensor(image, mstype.float32)
            label = Tensor(label, mstype.int32)
            _, scrutinizer_out, _, _ = self.model(image_data)
            result_label, _ = ops.ArgMaxWithValue(1)(scrutinizer_out)
            success_num = success_num + sum((result_label == label).asnumpy())
            total_num = total_num + float(image_data.shape[0])
        accuracy = round(success_num / total_num, 3)
        self.print('cur epoch {},top1 accuracy {}.'.format(cb_params.cur_epoch_num, accuracy))
        self.model.set_train(True)
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            save_checkpoint(self.model, self.save_path)
