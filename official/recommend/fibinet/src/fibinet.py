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
"""FiBiNet model"""
import itertools
from mindspore import nn, context
from mindspore import Parameter, ParameterTuple
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.nn import Dropout
from mindspore.nn.optim import Adam, FTRL, LazyAdam
from mindspore.common.initializer import Uniform, initializer
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size
ms_type = mstype.float32


def init_method(method, shape, name, max_val=1.0):
    """parameter init method"""
    if method in ['uniform']:
        params = Parameter(initializer(Uniform(max_val), shape, ms_type), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, ms_type), name=name)
    elif method == "normal":
        params = Parameter(initializer("normal", shape, ms_type), name=name)
    return params


def init_var_dict(init_args, in_vars):  # [('Wide_b', [1], config.emb_init)]
    """var init function"""
    var_map = {}
    _, _max_val = init_args
    for _, item in enumerate(in_vars):
        key, shape, method = item
        if key not in var_map.keys():
            if method in ['random', 'uniform']:
                var_map[key] = Parameter(initializer(Uniform(_max_val), shape, ms_type), name=key)
            elif method == "one":
                var_map[key] = Parameter(initializer("ones", shape, ms_type), name=key)
            elif method == "zero":
                var_map[key] = Parameter(initializer("zeros", shape, ms_type), name=key)
            elif method == 'normal':
                var_map[key] = Parameter(initializer("normal", shape, ms_type), name=key)
    return var_map


class DenseLayer(nn.Cell):
    """
    Dense Layer for Deep Layer of FiBiNet Model;
    Containing: activation, matmul, bias_add;
    """

    def __init__(self, input_dim, output_dim, weight_bias_init, act_str,
                 keep_prob=0.5, use_activation=True, convert_dtype=True, drop_out=False):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(
            weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = ops.MatMul(transpose_b=False)
        self.bias_add = ops.BiasAdd()
        self.cast = ops.Cast()
        self.dropout = Dropout(keep_prob=keep_prob)
        self.use_activation = use_activation
        self.convert_dtype = convert_dtype
        self.drop_out = drop_out

    def _init_activation(self, act_str):
        act_str = act_str.lower()
        if act_str == "relu":
            act_func = ops.ReLU()
        elif act_str == "sigmoid":
            act_func = ops.Sigmoid()
        elif act_str == "tanh":
            act_func = ops.Tanh()
        return act_func

    def construct(self, x):
        """Construct Dense layer"""
        if self.training and self.drop_out:
            x = self.dropout(x)
        if self.convert_dtype:
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            bias = self.cast(self.bias, mstype.float16)
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.use_activation:
                wx = self.act_func(wx)
            wx = self.cast(wx, mstype.float32)
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.use_activation:
                wx = self.act_func(wx)
        return wx


class SENETLayer(nn.Cell):
    """SENETLayer used in FiBiNET. Essentially a weighted process of input sparse features.
      Input shape
        - A 3D tensor with shape: ``(batch_size,field_size,emb_dim)``.
      Output shape
        - A 3D tensor with shape: ``(batch_size,field_size,emb_dim)``.
      Arguments
        - field_size: Positive integer, number of features.
        - reduction_ratio: Positive integer, the degree to which the result obtained by Squeeze process is
                                compressed in the first fully-connected layer of Excitation process, the
                                dimension will be reduced from "field_size" to "field_size/reduction_ratio" here.
        - seed: Random integer, can be any integer.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
           Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, field_size, reduction_ratio=3, seed=1024):
        super(SENETLayer, self).__init__()
        seed = seed
        reduced_size = max(1, field_size // reduction_ratio)
        self.mean_pooling = ops.ReduceMean(keep_dims=False)
        self.excitation = nn.SequentialCell(
            [nn.Dense(field_size, reduced_size, has_bias=False),
             nn.ReLU(),
             nn.Dense(reduced_size, field_size, has_bias=False),
             nn.ReLU()]
        )

    def construct(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        squeeze = self.mean_pooling(inputs, -1)
        excitation = self.excitation(squeeze)
        re_weight = ops.mul(inputs, ops.ExpandDims()(excitation, 2))

        return re_weight


class BilinearInteraction(nn.Cell):
    """BilinearInteraction Layer used in FiBiNET, generate interaction terms by weighting a feature first
            and then get the Hadamard product of the weighted feature and another feature.
      Input:
        - A 3D tensor with shape: ``(batch_size,field_size, emb_dim)``.
      Output:
        - A 3D tensor with shape: ``(batch_size,field_size*(field_size-1)/2, emb_dim)``.
      Arguments
        - field_size: Positive integer, number of features.
        - emb_dim: Positive integer, embedding size of sparse features.
        - bilinear_type: String, types of bilinear functions used, it should be one of ['all', 'each', 'interaction'].
        - seed: Integer, random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
           Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, field_size, emb_dim, bilinear_type="all", seed=1024):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        seed = seed
        self.split = ops.Split()
        self.comb = list(itertools.combinations(range(field_size), 2))
        self.comb0 = [i[0] for i in self.comb]
        self.comb1 = [i[1] for i in self.comb]
        self.flatten = ops.Flatten()

        if bilinear_type not in ["all", "each", "interaction"]:
            raise NotImplementedError

        self.selcet_index = [self.bilinear_type == "all",
                             self.bilinear_type == "each",
                             self.bilinear_type == "interaction"].index(1)

        self.bilinear_all = nn.Dense(len(self.comb0)*emb_dim, len(self.comb0)*emb_dim, has_bias=False)

        self.bilinear_each = nn.CellList()
        for _ in range(field_size):
            self.bilinear_each.append(nn.Dense(emb_dim, emb_dim, has_bias=False))

        self.bilinear_interaction = nn.CellList()
        for _ in range(len(self.comb)):
            self.bilinear_interaction.append(nn.Dense(emb_dim, emb_dim, has_bias=False))


    def construct(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        bilinear_in = ops.Split(1, inputs.shape[1])(inputs)

        type_all = self.bilinear_all(self.flatten(inputs[:, self.comb0, :])).reshape(
            (inputs.shape[0], len(self.comb0), inputs.shape[2]))
        type_all = ops.Mul()(type_all, inputs[:, self.comb1, :])

        type_each = []
        for index in self.comb0:
            type_each.append(self.bilinear_each[index](bilinear_in[index]))
        type_each = ops.Mul()(ops.Concat(1)(type_each), inputs[:, self.comb1, :])

        type_interaction = []
        for index, bilinear in zip(self.comb0, self.bilinear_interaction):
            type_interaction.append(bilinear(bilinear_in[index]))
        type_interaction = ops.Mul()(ops.Concat(1)(type_interaction), inputs[:, self.comb1, :])

        interaction = [type_all, type_each, type_interaction][self.selcet_index]

        return interaction


class FiBiNetModel(nn.Cell):
    """
        From paper: " FiBiNET: Combining Feature Importance and Bilinear feature Interaction
                      for Click-Through Rate Prediction "
        Args:
            config (Class): The default config of FiBiNet
    """
    def __init__(self, config):
        super(FiBiNetModel, self).__init__()

        self.field_size = config.field_size
        self.linear_features_num = config.dense_dim
        emb_dim = config.emb_dim

        # intercept of wide part
        var_map = init_var_dict(config.init_args, [('Wide_b', [1], config.emb_init)])
        self.wide_b = var_map["Wide_b"]

        self.SE = SENETLayer(config.slot_dim, config.reduction_ratio, config.seed)
        self.Bilinear = BilinearInteraction(config.slot_dim, emb_dim, config.bilinear_type, config.seed)
        deep_input_dims = 2 * len(list(itertools.combinations(range(config.slot_dim), 2))) * emb_dim + \
                          self.linear_features_num
        all_dim_list = [deep_input_dims] + config.deep_layer_dim + [1]
        self.dense_layer_1 = DenseLayer(all_dim_list[0], all_dim_list[1], config.weight_bias_init,
                                        config.deep_layer_act, convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_2 = DenseLayer(all_dim_list[1], all_dim_list[2], config.weight_bias_init,
                                        config.deep_layer_act, convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_3 = DenseLayer(all_dim_list[2], all_dim_list[3], config.weight_bias_init,
                                        config.deep_layer_act, convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_4 = DenseLayer(all_dim_list[3], all_dim_list[4], config.weight_bias_init,
                                        config.deep_layer_act, convert_dtype=True, drop_out=config.dropout_flag,
                                        use_activation=False)

        self.mul = ops.Mul()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.reshape = ops.Reshape()
        self.flatten = ops.Flatten()
        sparse = config.sparse
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        self.batch_size = config.batch_size

        if is_auto_parallel:
            self.batch_size = self.batch_size * get_group_size()

        if is_auto_parallel and sparse and not config.field_slice:
            target = 'DEVICE'
            self.wide_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, 1, target=target,
                                                           slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE)
            if config.deep_table_slice_mode == "column_slice":
                self.deep_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, emb_dim, target=target,
                                                               slice_mode=nn.EmbeddingLookup.TABLE_COLUMN_SLICE)
                self.dense_layer_1.dropout.dropout.shard(((1, get_group_size()),))
                self.dense_layer_1.matmul.shard(((1, get_group_size()), (get_group_size(), 1)))
                self.dense_layer_1.matmul.add_prim_attr("field_size", self.field_size)
                self.mul.shard(((1, 1, get_group_size()), (1, 1, 1)))
                self.reshape.add_prim_attr("skip_redistribution", True)
            else:
                self.deep_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, emb_dim, target=target,
                                                               slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE)
            self.reduce_sum.add_prim_attr("cross_batch", True)
            self.embedding_table = self.deep_embeddinglookup.embedding_table

        else:
            self.deep_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, emb_dim,
                                                           target='DEVICE', sparse=sparse,
                                                           vocab_cache_size=config.vocab_cache_size)
            self.wide_embeddinglookup = nn.EmbeddingLookup(config.vocab_size, 1,
                                                           target='DEVICE', sparse=sparse,
                                                           vocab_cache_size=config.vocab_cache_size)
            self.embedding_table = self.deep_embeddinglookup.embedding_table


    def construct(self, id_hldr, wt_hldr):
        """
        Args:
            id_hldr: batch ids; # ids of all levels (feaures after one-hot) in this batch
            wt_hldr: batch weights;  # batch weights is fixed variables in Criteo dataset, which means each sample must be weighted
        """

        # generate input
        mask = self.reshape(wt_hldr, (self.batch_size, self.field_size, 1))

        wide_id_input = self.wide_embeddinglookup(id_hldr)
        linear_features = wide_id_input[:, :self.linear_features_num, :]
        wide_input = self.mul(wide_id_input, mask)

        deep_id_emb = self.deep_embeddinglookup(id_hldr)
        deep_input_weighted = self.mul(deep_id_emb, mask)
        embedded_sparse_features = deep_input_weighted[:, self.linear_features_num:, :]
        senet_output = self.SE(embedded_sparse_features)
        senet_bilinear_out = self.Bilinear(senet_output)
        bilinear_output = self.Bilinear(embedded_sparse_features)
        interaction_input = self.flatten(ops.Concat(1)((senet_bilinear_out, bilinear_output)))

        # generate output
        wide_out = self.reshape(self.reduce_sum(wide_input, 1) + self.wide_b, (-1, 1))

        deep_in = ops.Concat(1)((interaction_input, self.flatten(linear_features)))
        deep_in = self.dense_layer_1(deep_in)
        deep_in = self.dense_layer_2(deep_in)
        deep_in = self.dense_layer_3(deep_in)
        deep_out = self.dense_layer_4(deep_in)

        out = wide_out + deep_out

        return out, self.embedding_table


class NetWithLossClass(nn.Cell):
    """"
    Provide FiBiNet training loss through network.
    Args:
        network (Cell): The training network
        config (Class): FiBiNet config
    """

    def __init__(self, network, config):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        sparse = config.sparse
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        self.no_l2loss = sparse
        self.network = network
        self.l2_coef = config.l2_coef
        self.loss = ops.SigmoidCrossEntropyWithLogits()
        self.square = ops.Square()
        self.reduceMean_false = ops.ReduceMean(keep_dims=False)
        if is_auto_parallel:
            self.reduceMean_false.add_prim_attr("cross_batch", True)
        self.reduceSum_false = ops.ReduceSum(keep_dims=False)

    def construct(self, batch_ids, batch_wts, label):
        """
        Construct NetWithLossClass
        """
        predict, embedding_table = self.network(batch_ids, batch_wts)
        log_loss = self.loss(predict, label)
        wide_loss = self.reduceMean_false(log_loss)
        if self.no_l2loss:
            deep_loss = wide_loss
        else:
            l2_loss_v = self.reduceSum_false(self.square(embedding_table)) / 2
            deep_loss = self.reduceMean_false(log_loss) + self.l2_coef * l2_loss_v

        return wide_loss, deep_loss


class IthOutputCell(nn.Cell):
    def __init__(self, network, output_index):
        super(IthOutputCell, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, x1, x2, x3):
        predict = self.network(x1, x2, x3)[self.output_index]
        return predict


class TrainStepWrap(nn.Cell):
    """
    Encapsulation class of FiBiNet network training.
    Append Adam and FTRL optimizers to the training network after that construct
    function can be called to create the backward graph.
    Args:
        network (Cell): The training network. Note that loss function should have been added.
        sens (Number): The adjust parameter. Default: 1024.0
    """

    def __init__(self, network, sens=1024.0, sparse=False, cache_enable=False):
        super(TrainStepWrap, self).__init__()
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        self.network = network
        self.network.set_train()
        self.trainable_params = network.trainable_params()
        weights_w = []
        weights_d = []
        for params in self.trainable_params:
            if 'wide' in params.name:
                weights_w.append(params)
            else:
                weights_d.append(params)
        self.weights_w = ParameterTuple(weights_w)
        self.weights_d = ParameterTuple(weights_d)

        if sparse and is_auto_parallel:
            self.optimizer_d = LazyAdam(
                self.weights_d, learning_rate=0.0001, eps=1e-8, loss_scale=sens)
            self.optimizer_w = FTRL(learning_rate=0.0001, params=self.weights_w,
                                    l1=1e-8, l2=1e-8, initial_accum=1.0, loss_scale=sens)
        else:
            self.optimizer_d = Adam(
                self.weights_d, learning_rate=0.0001, eps=1e-8, loss_scale=sens)
            self.optimizer_w = FTRL(learning_rate=0.0001, params=self.weights_w,
                                    l1=1e-8, l2=1e-8, initial_accum=1.0, loss_scale=sens)
        self.hyper_map = ops.HyperMap()
        self.grad_w = ops.GradOperation(get_by_list=True,
                                        sens_param=True)
        self.grad_d = ops.GradOperation(get_by_list=True,
                                        sens_param=True)
        self.sens = sens
        self.loss_net_w = IthOutputCell(network, output_index=0)
        self.loss_net_d = IthOutputCell(network, output_index=1)
        self.loss_net_w.set_grad()
        self.loss_net_d.set_grad()

        self.reducer_flag = False
        self.grad_reducer_w = None
        self.grad_reducer_d = None
        self.reducer_flag = parallel_mode in (ParallelMode.DATA_PARALLEL,
                                              ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer_w = DistributedGradReducer(self.optimizer_w.parameters, mean, degree)
            self.grad_reducer_d = DistributedGradReducer(self.optimizer_d.parameters, mean, degree)

    def construct(self, batch_ids, batch_wts, label):
        """
        Construct FiBiNetModel model
        """
        weights_w = self.weights_w
        weights_d = self.weights_d
        loss_w, loss_d = self.network(batch_ids, batch_wts, label)
        sens_w = ops.Fill()(ops.DType()(loss_w), ops.Shape()(loss_w), self.sens)
        sens_d = ops.Fill()(ops.DType()(loss_d), ops.Shape()(loss_d), self.sens)
        grads_w = self.grad_w(self.loss_net_w, weights_w)(batch_ids, batch_wts,
                                                          label, sens_w)
        grads_d = self.grad_d(self.loss_net_d, weights_d)(batch_ids, batch_wts,
                                                          label, sens_d)
        if self.reducer_flag:
            grads_w = self.grad_reducer_w(grads_w)
            grads_d = self.grad_reducer_d(grads_d)
        return ops.depend(loss_w, self.optimizer_w(grads_w)), ops.depend(loss_d, self.optimizer_d(grads_d))


class PredictWithSigmoid(nn.Cell):
    """
    Predict definition
    """

    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__()
        self.network = network
        self.sigmoid = ops.Sigmoid()
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        full_batch = context.get_auto_parallel_context("full_batch")
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if is_auto_parallel and full_batch:
            self.sigmoid.shard(((1, 1),))

    def construct(self, batch_ids, batch_wts, labels):
        logits, _, = self.network(batch_ids, batch_wts)
        pred_probs = self.sigmoid(logits)
        return logits, pred_probs, labels
