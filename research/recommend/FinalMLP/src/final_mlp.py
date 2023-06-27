# Copyright 2023 Huawei Technologies Co., Ltd
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

import numpy as np
from sklearn.metrics import roc_auc_score
import mindspore
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn.optim import Adam
from mindspore.nn.metrics import Metric
from mindspore import nn, Tensor, ParameterTuple, Parameter
from mindspore.common.initializer import Uniform, initializer
from mindspore.context import ParallelMode, get_auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


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
                                                astype(dtype=np_type)
                                                ), name=key
                                         )
    return var_map


class DenseLayer(nn.Cell):
    """
    Dense Layer for Deep Layer of WideDeep Model;
    Containing: activation, matmul, bias_add;
    """

    def __init__(self, input_dim, output_dim, weight_bias_init=None, act_str='relu',
                 scale_coef=1.0, convert_dtype=True, use_act=True
                 ):
        super(DenseLayer, self).__init__()
        if not weight_bias_init:
            weight_bias_init = ('uniform', 'zero')
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self.get_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.scale_coef = scale_coef
        self.convert_dtype = convert_dtype
        self.use_act = use_act

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

    def get_activation(self, act_str):
        """Init activation function"""
        act_str = act_str.lower()
        if act_str == "relu":
            act_func = P.ReLU()
        elif act_str == "sigmoid":
            act_func = P.Sigmoid()
        elif act_str == "tanh":
            act_func = P.Tanh()
        return act_func


class MlpBlock(nn.Cell):
    """
    MlpBlock module
    """
    def __init__(self,
                 input_dim,
                 hidden_units=None,
                 hidden_activations="relu",
                 output_dim=None,
                 output_activation=None,
                 dropout_rates=0.0,
                 batch_norm=False,
                 norm_before_activation=True):
        super(MlpBlock, self).__init__()
        dense_layers = []
        hidden_activations = nn.get_activation(hidden_activations)
        if hidden_units is None:
            hidden_units = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)

        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(DenseLayer(hidden_units[idx],
                                           hidden_units[idx + 1]))
            if norm_before_activation:
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if not norm_before_activation:
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(keep_prob=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(DenseLayer(hidden_units[-1],
                                           output_dim))
        if output_activation is not None:
            dense_layers.append(nn.get_activation(output_activation))
        self.mlp = nn.SequentialCell(dense_layers)  # * used to unpack list

    def construct(self, inputs):
        return self.mlp(inputs)


class FeatureSelection(nn.Cell):
    """
    FeatureSelection module
    """
    def __init__(self, batch_size, feature_dim, embedding_dim,
                 fs_hidden_units=None, fs1_context=None, fs2_context=None):
        super(FeatureSelection, self).__init__()
        if fs_hidden_units is None:
            fs_hidden_units = []
        if fs1_context is None:
            fs1_context = []
        if fs2_context is None:
            fs2_context = []
        self.batch_size = batch_size
        self.fs1_context = fs1_context
        self.fs2_context = fs2_context
        if not fs1_context:
            self.fs1_ctx_bias = initializer('zeros', [1, embedding_dim], mstype.float32)
        else:
            self.fs1_ctx_emb = None
        if not fs2_context:
            self.fs2_ctx_bias = initializer('zeros', [1, embedding_dim], mstype.float32)
        else:
            self.fs2_ctx_emb = None
        self.fs1_gate = MlpBlock(input_dim=embedding_dim * max(1, len(fs1_context)),
                                 output_dim=feature_dim,
                                 hidden_units=fs_hidden_units,
                                 hidden_activations="relu",
                                 output_activation="sigmoid",
                                 batch_norm=False
                                 )
        self.fs2_gate = MlpBlock(input_dim=embedding_dim * max(1, len(fs2_context)),
                                 output_dim=feature_dim,
                                 hidden_units=fs_hidden_units,
                                 hidden_activations="relu",
                                 output_activation="sigmoid",
                                 batch_norm=False
                                 )

    def construct(self, flat_emb):
        fs1_input = mindspore.numpy.tile(self.fs1_ctx_bias, [self.batch_size, 1])  # self.fs1_ctx_emb
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1
        fs2_input = mindspore.numpy.tile(self.fs2_ctx_bias, [self.batch_size, 1])  # self.fs2_ctx_emb
        gt2 = self.fs2_gate(fs2_input) * 2
        feature2 = flat_emb * gt2
        return feature1, feature2


class InteractionAggregation(nn.Cell):
    """
    InteractionAggregation module
    """
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = DenseLayer(x_dim, output_dim)
        self.w_y = DenseLayer(y_dim, output_dim)
        self.w_xy = initializer('xavier_uniform', [self.num_heads, self.head_x_dim, self.head_y_dim * self.output_dim])
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze()

    def construct(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = mindspore.ops.bmm(
            mindspore.ops.bmm(head_x.expand_dims(2),
                              mindspore.numpy.tile(self.w_xy.view(1, self.num_heads, self.head_x_dim, -1),
                                                   [head_x.shape[0], 1, 1, 1])
                             ).view(-1, self.num_heads, self.output_dim, self.head_y_dim),
            head_y.expand_dims(-1)
        ).squeeze(-1)
        output += xy.sum(axis=1)
        return output


class FinalMLP(nn.Cell):
    """
    FinalMLP model definition
    """

    def __init__(self, config):
        super(FinalMLP, self).__init__()
        self.batch_size = config.batch_size
        self.field_size = config.data_field_size
        self.dense_field_size = config.get("dense_field_size", 0)
        self.vocab_size = config.data_vocab_size
        self.emb_dim = config.data_emb_dim

        self.mlp1_hidden_units = config.get("mlp1_hidden_units", [64, 64, 64])
        self.mlp1_hidden_activations = config.get("mlp1_hidden_activations", "relu")
        self.mlp1_dropout = config.get("mlp1_dropout", 0)
        self.mlp1_batch_norm = config.get("mlp1_batch_norm", False)
        self.mlp2_hidden_units = config.get("mlp2_hidden_units", [64, 64, 64])
        self.mlp2_hidden_activations = config.get("mlp2_hidden_activations", "relu")
        self.mlp2_dropout = config.get("mlp2_dropout", 0)
        self.mlp2_batch_norm = config.get("mlp2_batch_norm", False)
        self.use_fs = config.get("use_fs", True)
        self.fs_hidden_units = config.get("fs_hidden_units", [64])
        self.fs1_context = config.get("fs1_context", [])
        self.fs2_context = config.get("fs2_context", [])
        self.num_heads = config.get("num_heads", 1)
        self.init_args = config.init_args
        init_acts = [('embedding', [self.vocab_size, self.emb_dim], 'normal')]
        var_map = init_var_dict(self.init_args, init_acts)
        self.embedding_layer = var_map.get("embedding", None)

        self.input_dims = self.field_size * self.emb_dim + self.dense_field_size

        self.mlp1 = MlpBlock(input_dim=self.input_dims,
                             output_dim=None,
                             hidden_units=self.mlp1_hidden_units,
                             hidden_activations=self.mlp1_hidden_activations,
                             output_activation=None,
                             dropout_rates=self.mlp1_dropout,
                             batch_norm=self.mlp1_batch_norm
                             )
        self.mlp2 = MlpBlock(input_dim=self.input_dims,
                             output_dim=None,
                             hidden_units=self.mlp2_hidden_units,
                             hidden_activations=self.mlp2_hidden_activations,
                             output_activation=None,
                             dropout_rates=self.mlp2_dropout,
                             batch_norm=self.mlp2_batch_norm
                             )
        if self.use_fs:
            self.fs_module = FeatureSelection(self.batch_size,
                                              self.input_dims,
                                              self.emb_dim,
                                              self.fs_hidden_units,
                                              self.fs1_context,
                                              self.fs2_context
                                              )
        self.fusion_module = InteractionAggregation(self.mlp1_hidden_units[-1],
                                                    self.mlp2_hidden_units[-1],
                                                    output_dim=1,
                                                    num_heads=self.num_heads)
        self.gatherv2 = P.Gather()
        self.mul = P.Mul()
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=1)
        self.matmul = P.MatMul(transpose_b=True)
        self.expand_dims = P.ExpandDims()
        self.sigmoid = P.Sigmoid()
        self.act_fun = P.Tanh()

    def construct(self, *inputs):
        """
        Args:
            id_hldr: batch ids;   [bs, field_size]
            wt_hldr: batch weights;   [bs, field_size]
        """
        if len(inputs) == 3:
            id_hldr, wt_hldr, _ = inputs
            dense_feat = None
        else:
            id_hldr, dense_feat, wt_hldr, _ = inputs
        mask = self.reshape(wt_hldr, (self.batch_size, self.field_size, 1))
        id_embs = self.gatherv2(self.embedding_layer, id_hldr, 0)
        embed = self.mul(id_embs, mask)

        flat_emb = self.reshape(embed, (-1, self.input_dims - self.dense_field_size))
        if self.dense_field_size > 0:
            flat_emb = self.concat((flat_emb, dense_feat))
        ###

        if self.use_fs:
            feat1, feat2 = self.fs_module(flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        return y_pred, self.embedding_layer


class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition
    """

    def __init__(self, network, l2_coef=1e-6):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.network = network
        self.l2_coef = l2_coef
        self.loss_fn = P.SigmoidCrossEntropyWithLogits()
        self.square = P.Square()
        self.reducemean_false = P.ReduceMean(keep_dims=False)
        self.reducesum_false = P.ReduceSum(keep_dims=False)

    def construct(self, *inputs):
        label = inputs[-1]
        predict, fm_id_embs = self.network(*inputs)
        loss = self.loss_fn(predict, label)
        mean_loss = self.reducemean_false(loss)
        l2_loss_v = self.reducesum_false(self.square(fm_id_embs))
        l2_loss_all = self.l2_coef * (l2_loss_v) * 0.5
        return mean_loss+l2_loss_all


class TrainStepWrap(nn.Cell):
    """
    TrainStepWrap definition
    """

    def __init__(self, network, train_config):
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = Adam(self.weights,
                              learning_rate=train_config.get("learning_rate", 0.001),
                              eps=train_config.get("epsilon", 1e-8),
                              loss_scale=train_config.get("loss_scale", 1.0),
                              weight_decay=train_config.get("weight_decay", 0.0))
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = train_config.get("loss_scale", 1.0)

        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(self.optimizer.parameters, mean, degree)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)  #
        grads = self.grad(self.network, weights)(*tuple(list(inputs) + [sens]))
        if self.reducer_flag:
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

    def construct(self, *inputs):
        label = inputs[-1]
        logits, _ = self.network(*inputs)
        pred_probs = self.sigmoid(logits)
        return logits, pred_probs, label


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

    def get_train_eval_net(self):
        final_net = FinalMLP(self.model_config)
        loss_net = NetWithLossClass(final_net, l2_coef=self.train_config.get("l2_coef", 0.0))
        train_net = TrainStepWrap(loss_net, self.train_config)
        eval_net = PredictWithSigmoid(final_net)
        return train_net, eval_net
