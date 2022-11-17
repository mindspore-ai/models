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
import os
import numpy as np
from mindspore import Model, Tensor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
import mindspore.common.dtype as mstype
from mindspore import nn as mnn
from mindspore import numpy as mnp
from mindspore import ops as mops
from mindspore.ops import operations as P
from mindspore.train.callback import Callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import _update_param, load_checkpoint, save_checkpoint
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.common import set_seed
from src.util import getCityLossWeight
from src.config import ms_train_data, repeat, save_path, ckpt_path, stage
from src.config import TrainConfig_1, TrainConfig_2, TrainConfig_3, TrainConfig_4
from src.config import weight_init, run_distribute, num_class
from src.model import ERFNet, Encoder_pred
from src.dataset import getCityScapesDataLoader_mindrecordDataset

set_seed(1)
# Pytorch NLLLoss + log_softmax
class SoftmaxCrossEntropyLoss(mnn.Cell):
    def __init__(self, num_cls, weight):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = mnn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.unsqueeze = mops.ExpandDims()
        self.get_size = mops.Size()
        self.exp = mops.Exp()
        self.pow = mops.Pow()
        self.weight = weight

    def construct(self, pred, labels):
        labels = self.cast(labels, mstype.int32)
        labels = self.reshape(labels, (-1,))
        pred = self.transpose(pred, (0, 2, 3, 1))
        pred = self.reshape(pred, (-1, self.num_cls))
        one_hot_labels = self.one_hot(labels, self.num_cls, self.on_value, self.off_value)
        pred = self.cast(pred, mstype.float32)
        num = self.get_size(labels)

        if self.weight is not None:
            weight = mnp.copy(self.weight)
            weight = self.cast(weight, mstype.float32)
            weight = self.unsqueeze(weight, 0)
            expand = mops.BroadcastTo(pred.shape)
            weight = expand(weight)
            weight_masked = weight[mnp.arange(num), labels]
            loss = self.ce(pred, one_hot_labels)
            loss = self.div(self.sum(loss * weight_masked), self.sum(weight_masked))
        else:
            loss = self.ce(pred, one_hot_labels)
            loss = self.div(self.sum(loss), num)
        return loss

class LossMonitor_mine(Callback):
    def __init__(self, per_print_times, learning_rate):
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.loss_list = []
        self.learning_rate = learning_rate

    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        print("epoch:%d lr: %s" % (cb_params.cur_epoch_num, \
            self.learning_rate[cb_params.cur_step_num]))

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            self.loss_list.append(loss)
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, \
                cur_step_in_epoch, loss))
            print("average loss is %s" % (np.mean(self.loss_list)))
            print()

    def epoch_end(self, run_context):
        self.loss_list = []

def attach(erfnet, encoder_pretrain):
    print("attach decoder.")
    encoder_trained_par = encoder_pretrain.parameters_dict()
    erfnet_par = erfnet.parameters_dict()
    for name, param_old in encoder_trained_par.items():
        if name.startswith("encoder"):
            erfnet_par[name].set_data(param_old)

def copy_param(net_new, net):
    print("copy param.")
    net_new_par = net_new.parameters_dict()
    net_par = net.parameters_dict()
    for name, param_old in net_par.items():
        if name.startswith("encoder"):
            _update_param(net_new_par[name], param_old)

def train(ckpt_path_, trainConfig_, rank_id, rank_size, stage_):
    print("stage:", stage_)
    save_prefix = "Encoder" if trainConfig_.encode else "ERFNet"
    if trainConfig_.epoch == 0:
        raise RuntimeError("?")

    if trainConfig_.encode:
        network = Encoder_pred(stage_, num_class, weight_init, run_distribute=run_distribute)
    else:
        network = ERFNet(stage_, num_class, weight_init, run_distribute=run_distribute)
    if not os.path.exists(ckpt_path_):
        print("load no ckpt file.")
    else:
        load_checkpoint(ckpt_file_name=ckpt_path_, net=network)
        print("load ckpt file:", ckpt_path_)

    # attach decoder
    if trainConfig_.attach_decoder:
        network_erfnet = ERFNet(stage_, num_class, weight_init, run_distribute=run_distribute)
        attach(network_erfnet, network)
        network = network_erfnet
    dataloader = getCityScapesDataLoader_mindrecordDataset(stage_, ms_train_data, 6, \
        trainConfig_.encode, trainConfig_.train_img_size, shuffle=True, aug=True, \
        rank_id=rank_id, global_size=rank_size, repeat=repeat)
    opt = mnn.Adam(network.trainable_params(), trainConfig_.lr, \
        weight_decay=1e-4, eps=1e-08)
    loss = SoftmaxCrossEntropyLoss(num_class, getCityLossWeight(trainConfig_.encode))

    loss_scale_manager = DynamicLossScaleManager()
    wrapper = Model(network, loss, opt, loss_scale_manager=loss_scale_manager, \
        keep_batchnorm_fp32=True, amp_level="O0")
    if rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps= \
                                    trainConfig_.epoch_num_save * dataloader.get_dataset_size(), \
                                    keep_checkpoint_max=5)
        saveModel_cb = ModelCheckpoint(prefix=save_prefix, directory= \
            save_path, config=config_ck)
        time_cb = TimeMonitor()
        call_backs = [time_cb, saveModel_cb, LossMonitor_mine(1, trainConfig_.lr.asnumpy())]
    else:
        call_backs = [LossMonitor_mine(1, trainConfig_.lr.asnumpy())]

    print("============== Starting {} Training ==============".format(save_prefix))
    wrapper.train(trainConfig_.epoch, dataloader, callbacks=call_backs, dataset_sink_mode=True)
    save_checkpoint(network, os.path.join(save_path, f"{save_prefix}_stage{stage_}.ckpt"))
    return network

if __name__ == "__main__":
    rank_id_ = 0
    rank_size_ = 1
    if run_distribute:
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        rank_id_ = get_rank()
        rank_size_ = get_group_size()
    else:
        # for single device: ascend/gpu
        context.set_context(device_id=int(os.environ["DEVICE_ID"]))

    trainConfig = {
        1: TrainConfig_1(),
        2: TrainConfig_2(),
        3: TrainConfig_3(),
        4: TrainConfig_4(),
    }

    network_ = train(ckpt_path, trainConfig[stage], rank_id=rank_id_, rank_size=rank_size_, stage_=stage)
