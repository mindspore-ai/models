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
""" test_training """

import os
import numpy as np
from sklearn.metrics import roc_auc_score

import mindspore
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn import Dropout
from mindspore.nn.optim import Adam
from mindspore.nn.metrics import Metric
from mindspore import nn, Tensor, ParameterTuple, Parameter
from mindspore.common.initializer import Uniform, initializer
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.context import ParallelMode, get_auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

from src.callback import EvalCallBack, LossCallBack

np_type = np.float32
ms_type = mstype.float32

class AUCMetric(Metric):
    """
    Metric method
    """
    def __init__(self):
        super(AUCMetric, self).__init__()
        self.pred_probs = []
        self.true_labels = []

    def clear(self):
        """Clear the internal evaluation result."""
        self.pred_probs = []
        self.true_labels = []

    def update(self, *inputs):
        batch_predict = inputs[1].asnumpy()
        batch_label = inputs[2].asnumpy()
        self.pred_probs.extend(batch_predict.flatten().tolist())
        self.true_labels.extend(batch_label.flatten().tolist())

    def eval(self):
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError('true_labels.size() is not equal to pred_probs.size()')
        auc = roc_auc_score(self.true_labels, self.pred_probs)
        return auc


def init_method(method, shape, name, max_val=1.0):
    if method in ['uniform']:
        params = Parameter(initializer(Uniform(max_val), shape, ms_type), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, ms_type), name=name)
    elif method == "normal":
        params = Parameter(Tensor(np.random.normal(loc=0.0, scale=0.01, size=shape).astype(dtype=np_type)), name=name)
    return params


def init_var_dict(init_args, var_list):
    """ init var with different methods. """
    var_map = {}
    _, max_val = init_args
    for i, _ in enumerate(var_list):
        key, shape, method = var_list[i]
        if key not in var_map.keys():
            if method in ['random', 'uniform']:
                var_map[key] = Parameter(initializer(Uniform(max_val), shape, ms_type), name=key)
            elif method == "one":
                var_map[key] = Parameter(initializer("ones", shape, ms_type), name=key)
            elif method == "zero":
                var_map[key] = Parameter(initializer("zeros", shape, ms_type), name=key)
            elif method == 'normal':
                var_map[key] = Parameter(Tensor(np.random.normal(loc=0.0, scale=0.01, size=shape).
                                                astype(dtype=np_type)), name=key)
    return var_map


class DenseLayer(nn.Cell):
    """
    Dense Layer for Deep Layer of WideDeep Model;
    Containing: activation, matmul, bias_add;
    Args:
    """
    def __init__(self, input_dim, output_dim, weight_bias_init, act_str, scale_coef=1.0, convert_dtype=True,
                 use_act=True):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.realDiv = P.RealDiv()
        self.scale_coef = scale_coef
        self.convert_dtype = convert_dtype
        self.use_act = use_act

    def _init_activation(self, act_str):
        """Init activation function"""
        act_str = act_str.lower()
        if act_str == "relu":
            act_func = P.ReLU()
        elif act_str == "sigmoid":
            act_func = P.Sigmoid()
        elif act_str == "tanh":
            act_func = P.Tanh()
        return act_func

    def construct(self, x):
        """Construct function"""
        if self.convert_dtype:
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            bias = self.cast(self.bias, mstype.float16)
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.use_act:
                wx = self.act_func(wx)
            wx = self.cast(wx, mstype.float32)
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.use_act:
                wx = self.act_func(wx)
        return wx


class FinalUnit(nn.Cell):
    def __init__(self, input_dim, output_dim, weight_bias_init, convert_dtype=True):
        super(FinalUnit, self).__init__()

        self.dense_layer_0 = DenseLayer(input_dim, output_dim, weight_bias_init,
                                        "tanh", 1.0,
                                        convert_dtype=convert_dtype, use_act=False)

        self.dense_layer_1 = DenseLayer(input_dim, output_dim, weight_bias_init,
                                        "tanh", 1.0,
                                        convert_dtype=convert_dtype, use_act=False)
        self.dense_layer_2 = DenseLayer(input_dim, output_dim, weight_bias_init,
                                        "tanh", 1.0,
                                        convert_dtype=convert_dtype, use_act=False)
        self.act_fun = P.Tanh()
        self.Mul = P.Mul()
        self.dropout = Dropout(keep_prob=0.9)
        self.BatchNorm = nn.BatchNorm1d(output_dim)

    def construct(self, x):
        x_0 = self.dense_layer_0(x)
        x_1 = self.final_interaction(x, x_0, self.dense_layer_1)
        x_2 = self.final_interaction(x, x_1, self.dense_layer_2)
        return x_2

    def final_interaction(self, x, x_i, layer):
        x_h = layer(x)
        x_h_act = self.act_fun(x_h)
        y = x_i + self.Mul(x_i, x_h_act)
        return y


class FinalBlock(nn.Cell):
    def __init__(self, dim_list, weight_bias_init, convert_dtype=True):
        super(FinalBlock, self).__init__()

        self.unit_0 = FinalUnit(dim_list[0], dim_list[1], weight_bias_init,
                                convert_dtype=convert_dtype)
        self.unit_1 = FinalUnit(dim_list[1], dim_list[2], weight_bias_init,
                                convert_dtype=convert_dtype)
        self.unit_2 = FinalUnit(dim_list[2], dim_list[3], weight_bias_init,
                                convert_dtype=convert_dtype)

        self.unit_3 = FinalUnit(dim_list[3], dim_list[4], weight_bias_init,
                                convert_dtype=convert_dtype)

        self.final_block_out = DenseLayer(dim_list[-1], 1, weight_bias_init,
                                          "tanh", 1.0,
                                          convert_dtype=convert_dtype, use_act=False)

    def construct(self, x):
        y0 = self.unit_0(x)
        y1 = self.unit_1(y0)
        y2 = self.unit_2(y1)
        y3 = self.unit_3(y2)
        y4 = self.final_block_out(y3)
        return y4


class FINALModel(nn.Cell):
    def __init__(self, config):
        super(FINALModel, self).__init__()

        self.batch_size = config.batch_size
        self.field_size = config.data_field_size
        self.vocab_size = config.data_vocab_size
        self.emb_dim = config.data_emb_dim

        self.block_orders = 3
        self.hidden_units = [256, 128, 64, 32]

        self.init_args = config.init_args
        self.weight_bias_init = config.weight_bias_init
        self.keep_prob = config.keep_prob

        convert_dtype = config.convert_dtype
        init_acts = [('embedding', [self.vocab_size, self.emb_dim], 'normal')]
        var_map = init_var_dict(self.init_args, init_acts)
        self.embedding_table = var_map["embedding"]

        self.input_dims = self.field_size * self.emb_dim
        self.all_dim_list = [self.input_dims] + self.hidden_units

        self.block_1 = FinalBlock(self.all_dim_list, self.weight_bias_init, convert_dtype=convert_dtype)
        self.block_2 = FinalBlock(self.all_dim_list, self.weight_bias_init, convert_dtype=convert_dtype)

        self.Gatherv2 = P.Gather()
        self.Mul = P.Mul()
        self.Reshape = P.Reshape()
        self.Concat = P.Concat(axis=1)
        self.MatMul = P.MatMul(transpose_b=True)
        self.ExpandDims = P.ExpandDims()
        self.SoftMax = P.Softmax()
        self.act_fun = P.Tanh()

    def construct(self, id_hldr, wt_hldr):
        """
        Args:
            id_hldr: batch ids;   [bs, field_size]
            wt_hldr: batch weights;   [bs, field_size]
        """

        mask = self.Reshape(wt_hldr, (self.batch_size, self.field_size, 1))
        id_embs = self.Gatherv2(self.embedding_table, id_hldr, 0)
        embed = self.Mul(id_embs, mask)
        deep_in = self.Reshape(embed, (-1, self.field_size * self.emb_dim))

        y_1 = self.block_1(deep_in)

        y_2 = self.block_2(deep_in)

        return y_1, y_2, self.embedding_table


class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition
    """
    def __init__(self, network, l2_coef=1e-6):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.network = network
        self.l2_coef = l2_coef
        self.Square = P.Square()
        self.sidmoid_fun = P.Sigmoid()
        self.ReduceMean_false = P.ReduceMean(keep_dims=False)
        self.ReduceSum_false = P.ReduceSum(keep_dims=False)

    def construct(self, batch_ids, batch_wts, label):
        y1, y2, fm_id_embs = self.network(batch_ids, batch_wts)
        predict = (y1 + y2)*0.5
        log_loss = self.loss(predict, label)
        proba = self.sidmoid_fun(predict)
        soft_label = mindspore.ops.stop_gradient(proba)
        aux_loss = self.loss(y1, soft_label) + self.loss(y2, soft_label)
        mean_aux_loss = self.ReduceMean_false(aux_loss)
        mean_log_loss = self.ReduceMean_false(log_loss)
        l2_loss_v = self.ReduceSum_false(self.Square(fm_id_embs))
        l2_loss_all = self.l2_coef * (l2_loss_v) * 0.5
        loss = mean_log_loss + l2_loss_all + mean_aux_loss
        return loss


class TrainStepWrap(nn.Cell):
    """
    TrainStepWrap definition
    """
    def __init__(self, network, lr, eps, loss_scale=1000.0):
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = Adam(self.weights, learning_rate=lr, eps=eps, loss_scale=loss_scale)
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = loss_scale

        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(self.optimizer.parameters, mean, degree)

    def construct(self, batch_ids, batch_wts, label):
        weights = self.weights
        loss = self.network(batch_ids, batch_wts, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens) #
        grads = self.grad(self.network, weights)(batch_ids, batch_wts, label, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


class PredictWithSigmoid(nn.Cell):
    """
    Predict method
    """
    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__(auto_prefix=False)
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, batch_ids, batch_wts, labels):
        y1, y2, _ = self.network(batch_ids, batch_wts)
        logits = 0.5 * (y1 + y2)
        pred_probs = self.sigmoid(logits)

        return logits, pred_probs, labels


class ModelBuilder():
    """
    Model builder for FINAL.

    Args:
        model_config (ModelConfig): Model configuration.
        train_config (TrainConfig): Train configuration.
    """
    def __init__(self, model_config, train_config):
        self.model_config = model_config
        self.train_config = train_config

    def get_callback_list(self, model=None, eval_dataset=None):
        """
        Get callbacks which contains checkpoint callback, eval callback and loss callback.

        Args:
            model (Cell): The network is added callback (default=None)
            eval_dataset (Dataset): Dataset for eval (default=None)
        """
        callback_list = []
        if self.train_config.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=self.train_config.save_checkpoint_steps,
                                         keep_checkpoint_max=self.train_config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix=self.train_config.ckpt_file_name_prefix,
                                      directory=self.train_config.output_path,
                                      config=config_ck)
            callback_list.append(ckpt_cb)
        if self.train_config.eval_callback:
            if model is None:
                raise RuntimeError("train_config.eval_callback is {}; get_callback_list() args model is {}".format(
                                        self.train_config.eval_callback, model))
            if eval_dataset is None:
                raise RuntimeError("train_config.eval_callback is {}; get_callback_list() args eval_dataset is {}".
                                   format(self.train_config.eval_callback, eval_dataset))
            auc_metric = AUCMetric()
            eval_callback = EvalCallBack(model, eval_dataset, auc_metric,
                                         eval_file_path=os.path.join(self.train_config.output_path,
                                                                     self.train_config.eval_file_name))
            callback_list.append(eval_callback)
        if self.train_config.loss_callback:
            loss_callback = LossCallBack(loss_file_path=os.path.join(self.train_config.output_path,
                                                                     self.train_config.loss_file_name))
            callback_list.append(loss_callback)
        if callback_list:
            return callback_list

        return None

    def get_train_eval_net(self):
        final_net = FINALModel(self.model_config)
        loss_net = NetWithLossClass(final_net, l2_coef=self.train_config.l2_coef)
        train_net = TrainStepWrap(loss_net, lr=self.train_config.learning_rate,
                                  eps=self.train_config.epsilon, loss_scale=self.train_config.loss_scale)
        eval_net = PredictWithSigmoid(final_net)
        return train_net, eval_net
