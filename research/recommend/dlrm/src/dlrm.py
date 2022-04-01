# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
test_training
"""
import os
import math
from typing import Sequence, List

import numpy as np
from sklearn.metrics import accuracy_score

import mindspore.common.dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn import Dropout
from mindspore.nn.optim import SGD, Adagrad
from mindspore.nn.metrics import Metric
from mindspore import context, nn, Tensor, ParameterTuple, Parameter
from mindspore.common.initializer import Uniform, initializer
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.context import ParallelMode, get_auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

from src.callback import EvalCallBack, LossCallBack

np_type = np.float32
ms_type = mstype.float32

# Use Accuracy as evaluation metric
class AccMetric(Metric):
    """Metric method
    """
    def __init__(self):
        super(AccMetric, self).__init__()
        self.pred_probs = []
        self.true_labels = []

    def clear(self):
        """Clear the internal evaluation results.
        """
        self.pred_probs.clear()
        self.true_labels.clear()

    def update(self, *inputs):
        """Update new predicted data.
        """
        batch_predict = inputs[0].asnumpy()
        batch_label = inputs[1].asnumpy()
        self.pred_probs.extend(batch_predict.flatten().tolist())
        self.true_labels.extend(batch_label.flatten().tolist())

    def eval(self):
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError('true_labels.size() is not equal to pred_probs.size()')
        preds = [1 if p >= 0.5 else 0 for p in self.pred_probs]
        acc = accuracy_score(self.true_labels, preds)
        return acc


def init_method(method, shape, name, max_val=1.0, std=0.01):
    """Init parameters.

    Return initialized parameters.

    Args:
        method (str): initialize method
        shape: parameter shape
        name: parameter name
        max_val (float): max value of uniform initialization
        std (float): the standard deviation of the normal distribution
    """
    if method in ['uniform']:
        params = Parameter(initializer(Uniform(max_val), shape, ms_type), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, ms_type), name=name)
    elif method == "normal":
        params = Parameter(Tensor(np.random.normal(loc=0.0, scale=std,
                                                   size=shape).astype(dtype=np_type)), name=name)
    return params


def init_var_dict(init_args, var_list):
    """Init var with different methods.
    """
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
    """Dense Layer.

    Attributes:
        weight_bias_init (str, str): weight and bias initialization type
        act_str: activation function name
        scale_coef:
        convert_dtype: whether to convert data type to float16
        use_act: whether to use activation
    """
    def __init__(self, input_dim, output_dim, weight_bias_init, act_str,
                 scale_coef=1.0, convert_dtype=False, use_act=True):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(weight_init, [input_dim, output_dim], name='weight',
                                  std=math.sqrt(2. / (input_dim + output_dim)))
        self.bias = init_method(bias_init, [output_dim], name='bias', std=math.sqrt(1. / output_dim))
        self.act_func = self._init_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()  # return a tensor with the new specified data type
        self.dropout = Dropout(keep_prob=1.0)
        self.mul = P.Mul()  # multiplies two tensors element-wise
        self.realDiv = P.RealDiv() # div element-wise
        self.scale_coef = scale_coef
        self.convert_dtype = convert_dtype
        self.use_act = use_act

    def _init_activation(self, act_str):
        """Init activation function
        """
        act_str = act_str.lower()
        if act_str == 'relu':
            act_func = P.ReLU()
        elif act_str == "sigmoid":
            act_func = P.Sigmoid()
        elif act_str == "tanh":
            act_func = P.Tanh()
        return act_func

    def construct(self, x):
        """Construct function.
        """
        x = self.dropout(x)
        if self.convert_dtype:
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            bias = self.cast(self.bias, mstype.float16)
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.use_act:
                wx = self.act_func(wx)
            wx = self.cast(wx, mstype.float32)  # convert back to float32
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.use_act:
                wx = self.act_func(wx)

        return wx

def create_mlp(layer_sizes, sigmoid_layer=False, convert_dtype=False):
    """Create a MLP by layer sizes

    Args:
        layer_sizes (list): all layer sizes of mlp
        sigmoid_layer (bool): whether the MLP has sigmoid activation in the last layer
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        if i == len(layer_sizes) - 2 and sigmoid_layer:
            layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], ('normal', 'normal'),
                                     'sigmoid', convert_dtype=convert_dtype, use_act=False))
        else:
            layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], ('normal', 'normal'),
                                     'relu', convert_dtype=convert_dtype))
    return nn.SequentialCell(layers)

class JointEmbedding(nn.Cell):
    """Buckle multiple one hot embedding together

    Multiple one hot embedding can be done as one embedding (indexing). Use nn.Embedding to deal with
    sparse wgrad before I fully customizing it.
    Args:
        categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
        embedding_dim (int): the size of each embedding vector
    """
    def __init__(self, categorical_feature_sizes: Sequence[int], embedding_dim: int, target, sparse):
        super().__init__()
        # cumsum [1, 2, 3] -> [1, 3, 6] - use a big Embedding table
        self.offsets = Tensor([0] + list(categorical_feature_sizes), dtype=mstype.int32).cumsum(0)
        rows = int(self.offsets.asnumpy()[-1])
        self.embedding = nn.EmbeddingLookup(rows, embedding_dim, target=target, sparse=sparse)

    def construct(self, categorical_inputs) -> List[Tensor]:
        """
        Args:
            categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]

        Returns:
            Tensor: embedding outputs in shape [batch, embedding_num, embedding_dim]
        """

        return [self.embedding(categorical_inputs + self.offsets[:-1])]

class MultiTableEmbeddings(nn.Cell):
    """Multiple Table Embedding Layer

    Attributes:
        categorical_feature_sizes: vocab size of every embedding table
        embedding_dim:
        target: CPU or GPU or NPU
    """
    def __init__(self, categorical_feature_sizes: Sequence[int], embedding_dim: int, target, sparse):
        super(MultiTableEmbeddings, self).__init__()

        embeddings = []
        # Each embedding table has size [num_features, embedding_dim]
        for num_features in categorical_feature_sizes:
            # NOTE: add initialization
            embedding = nn.Embedding(num_features, embedding_dim, embedding_table=Uniform(np.sqrt(1. / num_features)))
            embeddings.append(embedding)

        self.embeddings = nn.CellList(embeddings)
        self.embedding_dim = embedding_dim

    def construct(self, categorical_inputs) -> List[Tensor]:
        '''
        Args:
            categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]

        Returns:
            Tensor: embedding outputs in shape [batch, embedding_num, embedding_dim]
        '''
        # embeddin_outputs will be a list of (26 in the cast of Criteo (dataset)) fetched embedding with shape
        # [batch_size, embedding_size]
        embedding_outputs = []
        for embedding_id, embedding in enumerate(self.embeddings):
            embedding_outputs.append(embedding(categorical_inputs[:, embedding_id]))

        return embedding_outputs

class DLRM(nn.Cell):
    """DLRM Architecture

    From paper: "Deep Learning Recommendation Model for Personalization and Recommendation Systems"
    Args:
        batch_size(int):  smaple_number of per step in training; (int, batch_size=128)
        cat_filed_size(int): input categorical filed number, or called id_feature number; (int, filed_size=26)
        num_filed_size(int): input numerical filed number, or called id_feature number; (int, filed_size=16)
        vocab_size(int):  id_feature vocab size, id dict size;  (int, vocab_size=200000)
        emb_dim(int):  id embedding vector dim, id mapped to embedding vector; (int, emb_dim=16)\
        bottom_mlp_args(list): Bottom MLP Layer args, only contained layer sizes;
        top_mlp_args (list): Top MLP layer args, only contained layer sizes now;
        init_args(list): init args for Parameter init; (list, init_args=[min, max, seeds])
        weight_bias_init(list): weight, bias init method for deep layers;
                                (list[str], weight_bias_init=['random', 'zero'])
        keep_prob(float): if dropout_flag is True, keep_prob rate to keep connect; (float, keep_prob=0.8)

        host_device_mix (bool): whether open host device mix, CPU & GPU mix
        field_slice (bool): whether manually set emebdding field_slice
        vocab_cache_size (int)
        interaction_op (str): interaction operation type.
        interaction_op (bool): whether interaction with itself.
    """
    def __init__(self, config):
        super(DLRM, self).__init__()

        self.batch_size = config.batch_size
        self.cat_field_size = config.slot_dim
        self.num_field_size = config.dense_dim
        self.vocab_size = config.data_vocab_size
        self.emb_dim = config.data_emb_dim
        self.keep_prob = config.keep_prob
        self.interaction_op = config.interaction_op
        convert_dtype = config.convert_dtype
        self.interaction_itself = config.interaction_itself
        self.embedding_type = config.embedding_type

        # operations
        self.concat = P.Concat(axis=1)
        self.bmm = P.BatchMatMul()

        # about parallel
        paralle_mode = context.get_auto_parallel_context('parallel_mode')
        is_auto_parallel = paralle_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if is_auto_parallel:
            self.batch_size = self.batch_size * get_group_size()

        # Architecture components
        bottom_mlp_layers = [self.num_field_size] + config.bottom_mlp_args
        self.bottom_mlp = create_mlp(bottom_mlp_layers, convert_dtype=convert_dtype)  # bottom MLP

        # Interaction Layer
        if self.interaction_op not in ['dot', 'cat']:
            raise NotImplementedError(f"unknown interaction op: {self.interaction_op}")

        if self.interaction_op == 'dot' and self.interaction_itself:
            num_interactions = (self.cat_field_size + 2) * (self.cat_field_size + 1) // 2 + self.emb_dim
        elif self.interaction_op == 'dot':
            num_interactions = (self.cat_field_size + 1) * (self.cat_field_size) // 2 + self.emb_dim
        elif self.interaction_op == 'cat':
            num_interactions = (self.cat_field_size + 1) * self.emb_dim

        ni, nj = self.cat_field_size + 1, self.cat_field_size + 1
        offset = 1 if self.interaction_itself else 0
        self.li = Tensor([i for i in range(ni) for j in range(i + offset)])
        self.lj = Tensor([j for i in range(nj) for j in range(i + offset)])

        # top mlp
        top_mlp_layers = [num_interactions] + config.top_mlp_args
        self.top_mlp = create_mlp(top_mlp_layers, sigmoid_layer=True, convert_dtype=convert_dtype)

        # embedding layer
        self.embeddinglookup = MultiTableEmbeddings(config.categorical_feature_sizes, self.emb_dim,
                                                    target=config.embedding_target, sparse=config.sparse)


    def interaction_features(self, numeracal_feature: Tensor, categorical_features: Tensor):
        """Interaction Layer
        """

        (batch_size, d) = numeracal_feature.shape
        if self.interaction_op == 'dot':
            # concatenate dense and sparse features
            if self.embedding_type == 'joint':
                _numeracal_feature = numeracal_feature.view(-1, 1, d)
            else:
                _numeracal_feature = numeracal_feature

            T = self.concat([_numeracal_feature] + categorical_features).view(batch_size, -1, d)
            # perform a dot product
            Z = self.bmm(T, T.transpose((0, 2, 1)))
            # append dense feature with the interactions (into a row vector)
            Zflat = Z[:, self.li, self.lj]
            # concatenate dense features and interactions
            R = self.concat([numeracal_feature, Zflat])
        else:
            # concatenate features into a row vector
            if self.embedding_type == 'joint':
                _numeracal_feature = numeracal_feature.view(-1, 1, d)
            else:
                _numeracal_feature = numeracal_feature
            R = self.concat([_numeracal_feature] + categorical_features).view(batch_size, -1)

        return R


    def construct(self, categorical_inputs: Tensor, numerical_input: Tensor):
        """
        Args:
            categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
        """
        bottom_mlp_output = self.bottom_mlp(numerical_input)
        bottom_output = self.embeddinglookup(categorical_inputs)

        interaction_output = self.interaction_features(bottom_mlp_output, bottom_output)

        output = self.top_mlp(interaction_output)

        return output

class NetWithLossClass(nn.Cell):
    """NetWithLossClass
    """
    def __init__(self, network, l2_coef=0):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.network = network
        self.ReduceMean_false = P.ReduceMean(keep_dims=False)

    def construct(self, batch_cats, batch_nums, label):
        predict = self.network(batch_cats, batch_nums)
        log_loss = self.loss(predict, label)
        mean_log_loss = self.ReduceMean_false(log_loss)
        loss = mean_log_loss
        return loss

class TrainStepWrap(nn.Cell):
    """Train Step Wrap
    """
    def __init__(self, network, lr, loss_scale=1000.0, optimizer='sgd'):
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        if optimizer == 'sgd':
            self.optimizer = SGD(self.weights, learning_rate=lr, loss_scale=loss_scale)
        elif optimizer == 'adagrad':
            self.optimizer = Adagrad(self.weights, learning_rate=lr, loss_scale=loss_scale)
        else:
            raise RuntimeError(f"cannot use optimizer: {optimizer}")

        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = loss_scale

        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = get_auto_parallel_context('parallel_mode')
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = get_auto_parallel_context('gradients_mean')
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(self.optimizer.parameters, mean, degree)

    def construct(self, batch_cats, batch_nums, label):
        """construct
        """
        weights = self.weights
        loss = self.network(batch_cats, batch_nums, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(batch_cats, batch_nums, label, sens)

        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


class PredictWithSigmoid(nn.Cell):
    """Predict method
    """
    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__(auto_prefix=False)
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, batch_cats, batch_nums, labels):
        logits = self.network(batch_cats, batch_nums)
        pred_probs = self.sigmoid(logits)

        return pred_probs, labels

class ModelBuilder():
    """Model builder for DLRM

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
                                      directory=self.train_config.ckpt_path, config=config_ck)
            callback_list.append(ckpt_cb)

        if self.train_config.eval_callback:
            if model is None:
                raise RuntimeError("train_config.eval_callback is {}; get_callback_list() args model is {}".format(
                                        self.train_config.eval_callback, model))
            if eval_dataset is None:
                raise RuntimeError("train_config.eval_callback is {}; get_callback_list() args eval_dataset is {}".
                                   format(self.train_config.eval_callback, eval_dataset))

            acc_metric = AccMetric()
            eval_callback = EvalCallBack(model, eval_dataset, acc_metric,
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
        dlrm_net = DLRM(self.model_config)
        loss_net = NetWithLossClass(dlrm_net)
        train_net = TrainStepWrap(loss_net, self.train_config.learning_rate,
                                  loss_scale=self.train_config.loss_scale, optimizer=self.train_config.optimizer)
        eval_net = PredictWithSigmoid(dlrm_net)
        return train_net, eval_net
