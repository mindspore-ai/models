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

"""eval callback"""

import os
import numpy as np
from mindspore import Tensor
from mindspore.train.callback import Callback
from src.dataset import create_ssd_dataset
from src.eval_utils import metrics
from src.box_utils import default_boxes
from src.config import config
from src.ssd import SsdInferWithDecoder

class EvalCallBack(Callback):
    """
            define the method of eval
            """
    def __init__(self, eval_dataset, net, eval_per_epoch, json_path):
        self.net = net
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.json_path = json_path
        self.best_top_mAP = 0
        self.best_top_epoch = 0

    def epoch_end(self, run_context):
        """eval detail."""
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch > 300:
            if cur_epoch % self.eval_per_epoch == 0:
                net = SsdInferWithDecoder(self.net, Tensor(default_boxes), config)
                net.set_train(False)
                pred_data = []
                for data in self.eval_dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
                    img_id = data['img_id']
                    img_np = data['image']
                    image_shape = data['image_shape']
                    output = net(Tensor(img_np))
                    for batch_idx in range(img_np.shape[0]):
                        pred_data.append({"boxes": output[0].asnumpy()[batch_idx],
                                          "box_scores": output[1].asnumpy()[batch_idx],
                                          "img_id": int(np.squeeze(img_id[batch_idx])),
                                          "image_shape": image_shape[batch_idx]})
                mAP = metrics(pred_data, self.json_path)
                if mAP > self.best_top_mAP:
                    self.best_top_mAP = mAP
                    self.best_top_epoch = cur_epoch
                print(f"mAP: {mAP}"+f"   best_top_mAP: {self.best_top_mAP}"+f"   best_top_epochs:{self.best_top_epoch}")
                net.set_train(True)

def eval_callback(val_data_url, model, json_path, batch_size, eval_per_epoch=1):
    """
        Entrance of validation method
        """
    val_data_list = []
    for i in range(8):
        val_data = os.path.join(val_data_url, "ssd_eval.mindrecord" + str(i))
        val_data_list.append(val_data)
    dataset = create_ssd_dataset(val_data_list, batch_size, repeat_num=1,
                                 is_training=False, use_multiprocessing=False)
    callback = EvalCallBack(dataset, model, eval_per_epoch, json_path)
    return callback
