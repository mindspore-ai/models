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
#################Create EvalCallBack  ########################
"""
import numpy as np
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.communication.management import get_rank
from mindspore import Tensor, save_checkpoint

from src.c3d_model import C3D
from src.model_utils.config import config
from src.dataset import classification_dataset

class EvalCallBack(Callback):
    """EvalCallBack"""
    def __init__(self, model, eval_per_epoch, epoch_per_eval, save_ckpt_path, train_batch_num):
        config.load_type = 'test'
        self.model = model
        self.rank = get_rank() if config.is_distributed else 0
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval
        self.save_ckpt_path = save_ckpt_path
        self.eval_dataset, self.eval_dataset_len = classification_dataset(config.batch_size, 1, shuffle=True,
                                                                          repeat_num=1, drop_remainder=True)
        self.best_ckpt = 0
        self.best_acc = 0
        self.train_batch_num = train_batch_num

    def epoch_end(self, run_context):
        """culculate acc"""
        network = C3D(config.num_classes)
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        save_ckpt_path = self.save_ckpt_path + str(self.rank) + '-' + str(cur_epoch) + '_' \
                         + str(self.train_batch_num) + '.ckpt'
        # pre_trained
        param_dict = load_checkpoint(save_ckpt_path)
        param_not_load = load_param_into_net(network, param_dict)
        batch_num = self.eval_dataset.get_dataset_size()
        print('ckpt:', save_ckpt_path)
        print('param_not_load', param_not_load)
        if cur_epoch % self.eval_per_epoch == 0:
            network.set_train(mode=False)
            acc_sum, sample_num = 0, 0
            for idnum, (input_data, label) in enumerate(self.eval_dataset):
                predictions = network(Tensor(input_data))
                predictions, label = predictions.asnumpy(), label.asnumpy()
                acc = np.sum(np.argmax(predictions, 1) == label[:, -1])
                batch_size = label.shape[0]
                acc_sum += acc
                sample_num += batch_size
                if idnum % 20 == 0:
                    print("setep: {}/{}, acc: {}".format(idnum + 1, batch_num, acc / batch_size))

            top_1 = acc_sum / sample_num
            print('eval result: top_1 {:.3f}%'.format(top_1 * 100))
            if self.best_acc < top_1:
                self.best_acc = top_1
                self.best_ckpt = cur_epoch
                best_ckpt_file = 'best_acc.ckpt'
                best_ckpt_file = self.save_ckpt_path + str(self.rank) + best_ckpt_file
                save_checkpoint(network, best_ckpt_file)
            print('best result: top_1 {:.3f}%'.format(self.best_acc * 100))
            print('best ckpt:{}'.format(self.best_ckpt))
