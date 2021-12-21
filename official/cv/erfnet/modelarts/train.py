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
from mindspore import ops as mops
from mindspore.ops import operations as P
from mindspore.train.callback import Callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.common.parameter import Parameter
from mindspore.train.serialization import _update_param, \
    _check_checkpoint_param, Validator, Checkpoint, tensor_to_np_type, tensor_to_ms_type
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import get_rank, get_group_size, init
from src.config import data_path, repeat, save_path, ckpt_path
from src.config import TrainConfig
from src.config import weight_init, run_distribute, num_class
from src.model import ERFNet
from src.dataset import getDataLoader_GeneratorDataset

# Pytorch NLLLoss + log_softmax
class SoftmaxCrossEntropyLoss(mnn.Cell):
    def __init__(self, num_cls):
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

    def construct(self, pred, labels):
        labels = self.cast(labels, mstype.int32)
        labels = self.reshape(labels, (-1,))
        pred = self.transpose(pred, (0, 2, 3, 1))
        pred = self.reshape(pred, (-1, self.num_cls))
        one_hot_labels = self.one_hot(
            labels, self.num_cls, self.on_value, self.off_value)
        pred = self.cast(pred, mstype.float32)
        num = self.get_size(labels)
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
        print("epoch:%d lr: %s" % (cb_params.cur_epoch_num,
                                   self.learning_rate[cb_params.cur_step_num]))

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            self.loss_list.append(loss)
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num,
                                                      cur_step_in_epoch, loss))
            print("average loss is %s" % (np.mean(self.loss_list)))
            print()

    def epoch_end(self, run_context):
        self.loss_list = []

def load_src_checkpoint(ckpt_file_name, net=None, strict_load=False,
                        filter_prefix=None, dec_key=None, dec_mode="AES-GCM"):
    ckpt_file_name, filter_prefix = _check_checkpoint_param(
        ckpt_file_name, filter_prefix)
    dec_key = Validator.check_isinstance(
        'dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)
    checkpoint_list = Checkpoint()

    try:
        if dec_key is None:
            with open(ckpt_file_name, "rb") as f:
                pb_content = f.read()
        else:
            pb_content = _decrypt(ckpt_file_name, dec_key,
                                  len(dec_key), dec_mode)
            if pb_content is None:
                raise ValueError
        checkpoint_list.ParseFromString(pb_content)
    except BaseException as e:
        raise ValueError(e.__str__())

    parameter_dict = {}
    try:
        param_data_list = []
        for element_id, element in enumerate(checkpoint_list.value):
            if filter_prefix is not None and _check_param_prefix(filter_prefix, element.tag):
                continue
            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            np_type = tensor_to_np_type[data_type]
            ms_type = tensor_to_ms_type[data_type]
            element_data = np.frombuffer(data, np_type)
            param_data_list.append(element_data)
            if (element_id == len(checkpoint_list.value) - 1) or \
                    (element.tag != checkpoint_list.value[element_id + 1].tag):
                param_data = np.concatenate((param_data_list), axis=0)
                param_data_list.clear()
                dims = element.tensor.dims
                if dims == [0]:
                    if 'Float' in data_type:
                        param_data = float(param_data[0])
                    elif 'Int' in data_type:
                        param_data = int(param_data[0])
                    parameter_dict[element.tag] = Parameter(
                        Tensor(param_data, ms_type), name=element.tag)
                elif dims == [1]:
                    parameter_dict[element.tag] = Parameter(
                        Tensor(param_data, ms_type), name=element.tag)
                else:
                    param_dim = []
                    for dim in dims:
                        param_dim.append(dim)
                    param_value = param_data.reshape(param_dim)
                    parameter_dict[element.tag] = Parameter(
                        Tensor(param_value, ms_type), name=element.tag)

    except BaseException as e:
        raise RuntimeError(e.__str__())

    if not parameter_dict:
        raise ValueError(
            f"The loaded parameter dict is empty after filtering, please check filter_prefix.")

    for _, param in net.parameters_and_names():
        if param.name in parameter_dict:
            new_param = parameter_dict[param.name]
            if not isinstance(new_param, Parameter):
                msg = ("Argument parameter_dict element should be a Parameter, but got {}.".format(
                    type(new_param)))
                raise TypeError(msg)
            if param.name.startswith("decoder.pred"):
                continue
            else:
                _update_param(param, new_param, strict_load)
        else:
            param_not_load.append(param.name)

def train(ckpt_path_, trainConfig_, rank_id, rank_size):
    save_prefix = "ERFNet"
    if trainConfig_.epoch == 0:
        raise RuntimeError("?")

    network = ERFNet(num_class, weight_init, run_distribute=run_distribute)
    if not os.path.exists(ckpt_path_):
        raise RuntimeError("load no ckpt file."+ckpt_path_)

    load_src_checkpoint(ckpt_file_name=ckpt_path_, net=network)
    print("load ckpt file:", ckpt_path_)

    dataloader = getDataLoader_GeneratorDataset(data_path, batch_size=6,
                                                height=trainConfig_.train_img_size, shuffle=True, aug=True,
                                                rank_id=rank_id, global_size=rank_size, repeat=repeat)

    opt = mnn.Adam(network.trainable_params(), trainConfig_.lr,
                   weight_decay=1e-4, eps=1e-08)
    loss = SoftmaxCrossEntropyLoss(num_class)

    loss_scale_manager = DynamicLossScaleManager()
    wrapper = Model(network, loss, opt, loss_scale_manager=loss_scale_manager,
                    keep_batchnorm_fp32=True)

    if rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=trainConfig_.epoch_num_save * dataloader.get_dataset_size(),
                                     keep_checkpoint_max=9999)
        saveModel_cb = ModelCheckpoint(
            prefix=save_prefix, directory=save_path, config=config_ck)
        call_backs = [saveModel_cb, LossMonitor_mine(
            1, trainConfig_.lr.asnumpy())]
    else:
        call_backs = [LossMonitor_mine(1, trainConfig_.lr.asnumpy())]

    print("============== Starting {} Training ==============".format(save_prefix))
    wrapper.train(trainConfig_.epoch, dataloader,
                  callbacks=call_backs, dataset_sink_mode=True)
    return network

if __name__ == '__main__':
    rank_id_ = 0
    rank_size_ = 1
    if run_distribute:
        context.set_auto_parallel_context(parameter_broadcast=True)
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        init()
        rank_id_ = get_rank()
        rank_size_ = get_group_size()

    network_ = train(ckpt_path, TrainConfig(),
                     rank_id=rank_id_, rank_size=rank_size_)
