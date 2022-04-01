# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import mindspore.common.dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn import Dropout
from mindspore.nn.optim import Adam
from mindspore.nn.metrics import Metric
from mindspore import nn, ParameterTuple, Parameter
from mindspore.common.initializer import Uniform, initializer, Normal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from .callback import EvalCallBack, LossCallBack


np_type = np.float32
ms_type = mstype.float32
ms_type_16 = mstype.float16


class AUCMetric(Metric):
    """AUC metric for EDCN model."""
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


def init_method(method, shape, name, max_val=0.01):
    """
    The method of init parameters.

    Args:
        method (str): The method uses to initialize parameter.
        shape (list): The shape of parameter.
        name (str): The name of parameter.
        max_val (float): Max value in parameter when uses 'random' or 'uniform' to initialize parameter.

    Returns:
        Parameter.
    """
    if method in ['random', 'uniform']:
        params = Parameter(initializer(Uniform(max_val), shape, ms_type), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, ms_type), name=name)
    elif method == "normal":
        params = Parameter(initializer(Normal(max_val), shape, ms_type), name=name)
    return params


def init_var_dict(init_args, values):
    """
    Init parameter.

    Args:
        init_args (list): Define max and min value of parameters.
        values (list): Define name, shape and init method of parameters.

    Returns:
        dict, a dict ot Parameter.
    """
    var_map = {}
    _, _max_val = init_args
    for key, shape, init_flag, data_type in values:
        if key not in var_map.keys():
            if init_flag in ['random', 'uniform']:
                var_map[key] = Parameter(initializer(Uniform(_max_val), shape, data_type), name=key)
            elif init_flag == "one":
                var_map[key] = Parameter(initializer("ones", shape, data_type), name=key)
            elif init_flag == "zero":
                var_map[key] = Parameter(initializer("zeros", shape, data_type), name=key)
            elif init_flag == 'normal':
                var_map[key] = Parameter(initializer(Normal(_max_val), shape, data_type), name=key)
    return var_map


class DenseLayer(nn.Cell):
    """
    Dense Layer for Deep Layer of EDCN Model;
    Containing: activation, matmul, bias_add;
    Args:
        input_dim (int): the shape of weight at 0-aixs;
        output_dim (int): the shape of weight at 1-aixs, and shape of bias
        weight_bias_init (list): weight and bias init method, "random", "uniform", "one", "zero", "normal";
        act_str (str): activation function method, "relu", "sigmoid", "tanh";
        keep_prob (float): Dropout Layer keep_prob_rate;
        scale_coef (float): input scale coefficient;
    """
    def __init__(self, input_dim, output_dim, weight_bias_init, act_str, keep_prob=0.9, scale_coef=1.0):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_str = act_str
        if self.act_str != "none":
            self.act_func = self._init_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.dropout = Dropout(keep_prob=keep_prob)
        self.mul = P.Mul()
        self.realDiv = P.RealDiv()
        self.scale_coef = scale_coef

    def _init_activation(self, act_str):
        act_str = act_str.lower()
        if act_str == "relu":
            act_func = P.ReLU()
        elif act_str == "sigmoid":
            act_func = P.Sigmoid()
        elif act_str == "tanh":
            act_func = P.Tanh()
        return act_func

    def construct(self, x):
        """Dense Layer for Deep Layer of EDCN Model."""
        if self.act_str != "none":
            x = self.act_func(x)
        if self.training:
            x = self.dropout(x)
        x = self.mul(x, self.scale_coef)
        x = self.cast(x, mstype.float16)
        weight = self.cast(self.weight, mstype.float16)
        wx = self.matmul(x, weight)
        wx = self.cast(wx, mstype.float32)
        wx = self.realDiv(wx, self.scale_coef)
        output = self.bias_add(wx, self.bias)
        return output


class EDCNModel(nn.Cell):
    """
    From paper: "Enhancing Explicit and Implicit Feature Interactions via Information Sharing
    for Parallel Deep CTR Models"

    Args:
        batch_size (int):  smaple_number of per step in training; (int, batch_size=128)
        filed_size (int):  input filed number, or called id_feature number; (int, filed_size=39)
        vocab_size (int):  id_feature vocab size, id dict size;  (int, vocab_size=200000)
        emb_dim (int):  id embedding vector dim, id mapped to embedding vector; (int, emb_dim=100)
        deep_layer_args (list):  Deep Layer args, layer_dim_list, layer_activator;
                             (int, deep_layer_args=[[100, 100, 100], "relu"])
        init_args (list): init args for Parameter init; (list, init_args=[min, max, seeds])
        weight_bias_init (list): weight, bias init method for deep layers;
                            (list[str], weight_bias_init=['random', 'zero'])
        keep_prob (float): if dropout_flag is True, keep_prob rate to keep connect; (float, keep_prob=0.8)
    """
    def __init__(self, config):
        super(EDCNModel, self).__init__()

        self.batch_size = config.batch_size
        self.field_size = config.data_field_size
        self.vocab_size = config.data_vocab_size
        self.emb_dim = config.data_emb_dim
        self.deep_layer_act = config.deep_layer_args
        self.num_cross_layer = config.num_cross_layer
        self.init_args = config.init_args
        self.weight_bias_init = config.weight_bias_init
        self.keep_prob = config.keep_prob
        self.temperature = config.temperature
        self.batch_norm = config.batch_norm
        self.input_dims = self.field_size * self.emb_dim
        init_acts = [('embedding', [self.vocab_size, self.emb_dim], 'normal', ms_type),
                     ('cross_w', [self.num_cross_layer, 1, self.input_dims], 'normal', ms_type),
                     ('cross_b', [self.num_cross_layer, 1, self.input_dims], 'normal', ms_type),
                     ('regulation_cross', [self.num_cross_layer, self.field_size], 'one', ms_type),
                     ('regulation_deep', [self.num_cross_layer, self.field_size], 'one', ms_type)
                     ]
        var_map = init_var_dict(self.init_args, init_acts)
        self.embedding_table = var_map["embedding"]
        self.cross_w = var_map["cross_w"]
        self.cross_b = var_map["cross_b"]
        self.regulation_cross = var_map["regulation_cross"]
        self.regulation_deep = var_map["regulation_deep"]

        # Deep Layers
        self.all_dim_list = [self.input_dims] * (self.num_cross_layer + 1) + [1]
        self.dense_layer_1 = DenseLayer(self.all_dim_list[0], self.all_dim_list[1],
                                        self.weight_bias_init, self.deep_layer_act, self.keep_prob)
        self.dense_layer_2 = DenseLayer(self.all_dim_list[1], self.all_dim_list[2],
                                        self.weight_bias_init, self.deep_layer_act, self.keep_prob)
        self.dense_layer_3 = DenseLayer(self.all_dim_list[2], self.all_dim_list[3],
                                        self.weight_bias_init, self.deep_layer_act, self.keep_prob)
        self.dense_layers = [self.dense_layer_1, self.dense_layer_2, self.dense_layer_3]
        self.dense_layer_final = DenseLayer(self.input_dims * 3, self.all_dim_list[4],
                                            self.weight_bias_init, "none", 1.0)

        self.Gatherv2 = P.Gather()
        self.Mul = P.Mul()
        self.Reshape = P.Reshape()
        self.Concat = P.Concat(axis=1)
        self.MatMul = P.MatMul(transpose_b=True)
        self.ExpandDims = P.ExpandDims()
        self.SoftMax = P.Softmax()
        self.BatchNorm_cross = nn.BatchNorm2d(1)
        self.BatchNorm_deep = nn.BatchNorm2d(1)

    def regulation(self, embed, layer, batch_norm=False):
        """
        Regulation module.
        Args:
            embed: field embedding;   [bs, field_size, embed_size]
            layer: interaction layer index;   int
            batch_norm: flag of batch_norm;   bool
        """
        regulation_unit_cross = self.SoftMax(self.regulation_cross[layer] / self.temperature)
        regulation_unit_deep = self.SoftMax(self.regulation_deep[layer] / self.temperature)
        regulation_unit_cross = self.ExpandDims(self.ExpandDims(regulation_unit_cross, 1), 0)
        regulation_unit_deep = self.ExpandDims(self.ExpandDims(regulation_unit_deep, 1), 0)
        embed_cross = self.Mul(embed, regulation_unit_cross)
        embed_deep = self.Mul(embed, regulation_unit_deep)
        embed_cross = self.ExpandDims(embed_cross, 1)
        embed_deep = self.ExpandDims(embed_deep, 1)
        if batch_norm: # the batch_norm operation is time-consuming
            embed_cross = self.BatchNorm_cross(embed_cross)
            embed_deep = self.BatchNorm_deep(embed_deep)
        embed_cross = self.Reshape(embed_cross, (-1, self.input_dims))
        embed_deep = self.Reshape(embed_deep, (-1, self.input_dims))

        return embed_cross, embed_deep

    def cross(self, x0, xl, layer):
        """
        Cross Net.
        Args:
            x0: field embedding of the first layer;   [bs, field_size * embed_size]
            x1: field embedding of the current layer;   [bs, field_size * embed_size]
            layer: interaction layer index;   int
        """
        xlw = self.MatMul(xl, self.cross_w[layer])
        xl = x0 * xlw + self.cross_b[layer] + xl
        return xl

    def construct(self, id_hldr, wt_hldr):
        """
        Args:
            id_hldr: batch ids;   [bs, field_size]
            wt_hldr: batch weights;   [bs, field_size]
        """
        mask = self.Reshape(wt_hldr, (self.batch_size, self.field_size, 1))
        id_embs = self.Gatherv2(self.embedding_table, id_hldr, 0)
        embed = self.Mul(id_embs, mask)

        # regulation module
        embed_cross, embed_deep = self.regulation(embed, 0, self.batch_norm)
        x0 = embed_cross
        xl = x0
        cross_o = self.cross(x0, xl, 0)
        deep_o = self.dense_layers[0](embed_deep)
        # bridge module - element-wise product
        bridge_o = cross_o * deep_o

        for i in range(1, self.num_cross_layer):
            bridge_o = self.Reshape(bridge_o, (-1, self.field_size, self.emb_dim))
            # regulation module
            embed_cross, embed_deep = self.regulation(bridge_o, i, self.batch_norm)
            cross_o = self.cross(x0, embed_cross, i)
            deep_o = self.dense_layers[i](embed_deep)
            # bridge module - element-wise product
            bridge_o = cross_o * deep_o

        prediction_embed = self.Concat((cross_o, deep_o, bridge_o))
        deep_out = self.dense_layer_final(prediction_embed)
        out = deep_out
        return out, self.embedding_table, self.cross_w, self.cross_b


class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition.
    """
    def __init__(self, network, l1_coef=1e-6, l2_coef=1e-6):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.network = network
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.Square = P.Square()
        self.Abs = P.Abs()
        self.ReduceMean_false = P.ReduceMean(keep_dims=False)
        self.ReduceSum_false = P.ReduceSum(keep_dims=False)

    def construct(self, batch_ids, batch_wts, label):
        """
        Construct NetWithLossClass
        """
        predict, id_embs, cross_w, cross_b = self.network(batch_ids, batch_wts)
        log_loss = self.loss(predict, label)
        mean_log_loss = self.ReduceMean_false(log_loss)
        l2_loss_v = self.ReduceSum_false(self.Square(id_embs))
        l1_cross_w_loss = self.ReduceSum_false(self.Abs(cross_w))
        l1_cross_b_loss = self.ReduceSum_false(self.Abs(cross_b))
        l1_loss_all = self.l1_coef * (l1_cross_w_loss + l1_cross_b_loss)
        l2_loss_all = self.l2_coef * (l2_loss_v) * 0.5
        loss = mean_log_loss + l2_loss_all + l1_loss_all
        return loss


class TrainStepWrap(nn.Cell):
    """
    TrainStepWrap definition
    """
    def __init__(self, network, lr=5e-8, eps=1e-8, loss_scale=1000.0):
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = Adam(self.weights, learning_rate=lr, eps=eps, loss_scale=loss_scale)
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = loss_scale

    def construct(self, batch_ids, batch_wts, label):
        weights = self.weights
        loss = self.network(batch_ids, batch_wts, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens) #
        grads = self.grad(self.network, weights)(batch_ids, batch_wts, label, sens)
        self.optimizer(grads)
        return loss


class PredictWithSigmoid(nn.Cell):
    """
    Eval model with sigmoid.
    """
    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__(auto_prefix=False)
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, batch_ids, batch_wts, labels):
        logits, _, _, _ = self.network(batch_ids, batch_wts)
        pred_probs = self.sigmoid(logits)

        return logits, pred_probs, labels


class ModelBuilder:
    """
    Model builder for EDCN.

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
            model (Cell): The network is added callback (default=None).
            eval_dataset (Dataset): Dataset for eval (default=None).
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
                raise RuntimeError("train_config.eval_callback is {}; get_callback_list() "
                                   "args eval_dataset is {}".format(self.train_config.eval_callback, eval_dataset))
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
        edcn_net = EDCNModel(self.model_config)
        loss_net = NetWithLossClass(edcn_net, l1_coef=self.train_config.l1_coef, l2_coef=self.train_config.l2_coef)
        train_net = TrainStepWrap(loss_net, lr=self.train_config.learning_rate,
                                  eps=self.train_config.epsilon,
                                  loss_scale=self.train_config.loss_scale)
        eval_net = PredictWithSigmoid(edcn_net)
        return train_net, eval_net
