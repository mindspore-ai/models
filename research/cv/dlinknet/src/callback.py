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

import time

from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback

from src.dinknet import DinkNet50
from src.dinknet import DinkNet34


class MyCallback(Callback):
    def __init__(self, log_file, weight_file_name, rank_label, device_num, show_step=False, learning_rate=2e-4,
                 model_name='dinknet34'):
        super(MyCallback, self).__init__()
        self.no_optim = 0
        self.train_epoch_best_loss = 100.
        self.tic_begin = time.time()
        self.log = log_file
        self.file_name = weight_file_name
        self.rank_label = rank_label
        self.device_num = device_num
        self.show_step = show_step
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.current_epoch_loss_sum = 0
        self.step_count = 0
        self.step_per_epoch = 0

    def begin(self, run_context):
        """Called once before the network executing."""

    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        print(self.rank_label + '* epoch ' + str(epoch_num) + ' start!')

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        train_step_loss = cb_params.net_outputs
        cur_time = str(int(time.time() - self.tic_begin))
        optimizer = cb_params.optimizer
        print('********')
        self.log.write('********' + "\n")
        print(self.rank_label + "epoch_end {} step {}, loss is {}, scale sense is {}, overflow is {}".format(
            epoch_num, step_num, train_step_loss[0].asnumpy(), train_step_loss[2].asnumpy(),
            train_step_loss[1].asnumpy()))
        self.log.write(self.rank_label + "epoch_end {} step {}, loss is {}, scale sense is {}, overflow is {}".format(
            epoch_num, step_num, train_step_loss[0].asnumpy(), train_step_loss[2].asnumpy(),
            train_step_loss[1].asnumpy()) + '\n')
        # compute train_epoch_loss
        train_epoch_loss = self.current_epoch_loss_sum / self.step_count
        if self.step_per_epoch == 0:
            self.step_per_epoch = self.step_count
        self.current_epoch_loss_sum = 0
        self.step_count = 0

        print(self.rank_label + 'epoch:' + str(epoch_num) + '    time:' + cur_time)
        print(self.rank_label + 'train_loss:' + str(train_epoch_loss))

        self.log.write('epoch:' + str(epoch_num) + '    time:' + cur_time + "\n")
        self.log.write('train_loss:' + str(train_epoch_loss) + "\n")

        if train_epoch_loss >= self.train_epoch_best_loss:
            self.no_optim += 1
        else:
            self.no_optim = 0
            self.train_epoch_best_loss = train_epoch_loss
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=self.file_name)
            print(self.rank_label + "Save the minimum train loss checkpoint, the loss is", train_epoch_loss)
            self.log.write(self.rank_label + "Save the minimum train loss checkpoint, the loss is "
                           + str(train_epoch_loss) + '\n')
        if self.no_optim > 6 and self.device_num == 1:
            print(self.rank_label + 'early stop at %d epoch (cause no_optim > 6)' % epoch_num)
            self.log.write('early stop at %d epoch (cause no_optim > 6)' % epoch_num + '\n')
            run_context.request_stop()
            return
        if self.no_optim > 3 * max(1, self.device_num / 2):
            if self.learning_rate < 5e-7 and self.device_num == 1:
                print(self.rank_label + 'early stop at %d epoch (cause cur_lr < 5e-7)' % epoch_num)
                self.log.write('early stop at %d epoch (cause cur_lr < 5e-7)' % epoch_num + '\n')
                run_context.request_stop()
                return
            if self.model_name == 'dinknet34':
                network = DinkNet34()
            else:
                network = DinkNet50()
            param_dict = load_checkpoint(self.file_name)
            load_param_into_net(network, param_dict)
            cb_params.train_network = network
            old_learning_rate = self.learning_rate
            self.learning_rate = self.learning_rate / 5.0
            optimizer.learning_rate.set_data(self.learning_rate)
            print(self.rank_label + 'update learning rate: ' + str(old_learning_rate) + ' -> '
                  + str(self.learning_rate))
            self.log.write('update learning rate: ' + str(old_learning_rate) + ' -> ' + str(self.learning_rate) + "\n")
            # reset no_optim to 0
            self.no_optim = 0

        self.log.flush()

    def step_begin(self, run_context):
        """Called before each step beginning."""

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        if self.step_per_epoch != 0:
            step_num %= self.step_per_epoch
        train_step_loss = cb_params.net_outputs
        if self.show_step:
            print(self.rank_label + "epoch {} step {}, loss is {}, scale sense is {}, overflow is {}".format(
                epoch_num, step_num, train_step_loss[0].asnumpy(), train_step_loss[2].asnumpy(),
                train_step_loss[1].asnumpy()))
        self.current_epoch_loss_sum += train_step_loss[0].asnumpy()
        self.step_count += 1

    def end(self, run_context):
        """Called once after network training."""
        self.log.close()
