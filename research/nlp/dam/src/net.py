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
# ===========================================================================
"""DAM Net"""
import pickle
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import context
import mindspore.ops.composite as C
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.context import ParallelMode
from mindspore.common.tensor import Tensor
from mindspore import ParameterTuple
from mindspore.parallel._utils import _get_device_num

import src.layers as layers
import src.utils as op


class DAMNet(nn.Cell):
    """
    Deep Attention Matching network

    Args:
        config: The config of the network.
        emb_init (str): The pre-trained word embeddings. Default: None.
        train_mode (bool): The mode of the network. Default: True.
    """
    def __init__(self, config, emb_init=None, is_emb_init=False):
        super(DAMNet, self).__init__()
        print("DAM Net")

        self.max_turn_num = config.max_turn_num
        self.max_turn_len = config.max_turn_len
        self.vocab_size = config.vocab_size
        self.emb_size = config.emb_size
        self.stack_num = config.stack_num
        self.channel1_dim = config.channel1_dim
        self.channel2_dim = config.channel2_dim
        self.is_positional = config.is_positional
        self.attention_type = config.attention_type
        self.is_layer_norm = config.is_layer_norm
        self.is_mask = config.is_mask

        self.is_emb_init = is_emb_init
        self.emb_init = emb_init

        self.cast = P.Cast()
        self.sqrt = P.Sqrt()

        if self.is_emb_init and (self.emb_init is not None):
            print('Loading emb_init: ', self.emb_init)
            word_emb = pickle.load(open(self.emb_init, 'rb'), encoding="bytes")
            word_emb = Tensor(word_emb, mstype.float32)
            self.embedding = nn.Embedding(vocab_size=self.vocab_size + 1, embedding_size=self.emb_size,
                                          embedding_table=word_emb)
        else:
            self.embedding = nn.Embedding(self.vocab_size + 1, self.emb_size)

        if self.is_positional and self.stack_num > 0:
            print('Use positional')
            self.positional_encoding_vector = op.PositionalEncodingVector(length=self.max_turn_len,
                                                                          channels=self.emb_size,
                                                                          max_timescale=10)
        self_blocks = []
        for _ in range(self.stack_num):
            self_block = layers.Block(in_dim=self.emb_size,
                                      attention_type=self.attention_type,
                                      is_layer_norm=self.is_layer_norm,
                                      is_mask=self.is_mask)
            self_blocks.append(self_block)
        self.self_blocks = nn.CellList(self_blocks)

        t_att_r_blocks = []
        for _ in range(self.stack_num + 1):
            t_att_r_block = layers.Block(in_dim=self.emb_size,
                                         attention_type=self.attention_type,
                                         is_layer_norm=self.is_layer_norm,
                                         is_mask=self.is_mask)
            t_att_r_blocks.append(t_att_r_block)
        self.t_att_r_blocks = nn.CellList(t_att_r_blocks)

        r_att_t_blocks = []
        for _ in range(self.stack_num + 1):
            r_att_t_block = layers.Block(in_dim=self.emb_size,
                                         attention_type=self.attention_type,
                                         is_layer_norm=self.is_layer_norm,
                                         is_mask=self.is_mask)
            r_att_t_blocks.append(r_att_t_block)
        self.r_att_t_blocks = nn.CellList(r_att_t_blocks)

        self.stack_1 = P.Stack(axis=1)
        self.stack_2 = P.Stack(axis=2)
        self.concat = P.Concat(axis=1)
        self.batch_matmul_tran_b = op.BatchMatMulCell(transpose_a=False, transpose_b=True)

        self.cnn_3d = layers.CNN3d(2 * (self.stack_num + 1), self.channel1_dim, self.channel2_dim)
        self.flatten = nn.Flatten()
        self.out_fc = nn.Dense(576, 1, weight_init=Tensor(op.orthogonal_init([1, 576]), mstype.float32),
                               bias_init='zeros')

    def construct(self, turns, every_turn_len, response, response_len, labels):
        """DAM compute graph"""
        # response part
        labels = labels
        Hr = self.embedding(response)
        if self.is_positional and self.stack_num > 0:
            Hr = self.positional_encoding_vector(Hr)
        Hr_stack = [Hr]
        for block in self.self_blocks:
            Hr = block(Hr, Hr, Hr, Q_lengths=response_len, K_lengths=response_len)
            Hr_stack.append(Hr)

        # context part
        sim_turns = []
        for i in range(self.max_turn_num):
            turn = turns[:, i]
            turn_len = every_turn_len[:, i]
            Hu = self.embedding(turn)
            if self.is_positional and self.stack_num > 0:
                Hu = self.positional_encoding_vector(Hu)
            Hu_stack = [Hu]
            for block in self.self_blocks:
                Hu = block(Hu, Hu, Hu, Q_lengths=turn_len, K_lengths=turn_len)
                Hu_stack.append(Hu)
            # cross-attention
            r_a_t_stack = []
            t_a_r_stack = []
            for index in range(self.stack_num + 1):
                t_a_r = self.t_att_r_blocks[index](Hu_stack[index], Hr_stack[index], Hr_stack[index],
                                                   Q_lengths=turn_len, K_lengths=response_len)
                r_a_t = self.r_att_t_blocks[index](Hr_stack[index], Hu_stack[index], Hu_stack[index],
                                                   Q_lengths=response_len, K_lengths=turn_len)
                t_a_r_stack.append(t_a_r)
                r_a_t_stack.append(r_a_t)

            hr = self.stack_1(Hr_stack)
            hu = self.stack_1(Hu_stack)
            r_a_t = self.stack_1(r_a_t_stack)
            t_a_r = self.stack_1(t_a_r_stack)
            r_a_t = self.concat((r_a_t, hr))
            t_a_r = self.concat((t_a_r, hu))

            sim = self.batch_matmul_tran_b(t_a_r, r_a_t) / self.sqrt(self.cast(200, mstype.float32))
            # sim shape: [batch_size, 2*(stack_num+1), max_turn_len, max_turn_len]

            sim_turns.append(sim)
        sim = self.stack_2(sim_turns)
        # sim shape: [batch_size, 2 * (stack_num+1), max_turn_num, max_turn_len, max_turn_len]
        final_info = self.cnn_3d(sim)
        x = self.flatten(final_info)
        logits = self.out_fc(x)
        return logits


class DAMNetWithLoss(nn.Cell):
    """Calculate loss"""
    def __init__(self, network):
        super(DAMNetWithLoss, self).__init__()
        self.loss = layers.DAMLoss(clip_value=10)
        self.network = network

    def construct(self, turns, every_turn_len, response, response_len, labels):
        """Calculate loss"""
        predict = self.network(turns, every_turn_len, response, response_len, labels)
        loss = self.loss(predict, labels)
        return loss


GRADIENT_CLIP_TYPE = 0
GRADIENT_CLIP_VALUE = 1.0
clip_grad = P.MultitypeFuncGraph("clip_grad")
@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """Grads Clip"""
    if clip_type not in (0, 1):
        return grad
    dt = P.dtype(grad)
    if clip_type == 0:
        new_grad = P.clip_by_value(grad, P.cast(P.tuple_to_array((-clip_value,)), dt),
                                   P.cast(P.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, P.cast(P.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """Reduction of Loss Scale"""
    return grad * P.Reciprocal()(scale)


class DAMTrainOneStepCell(nn.Cell):
    """
    Encapsulation class of DAM network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        enable_clip_grad(bool): Whether apply grads clip before optimizer. Default: True
    """
    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True):
        super(DAMTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=False)
        self.sens = sens
        self.enable_clip_grad = enable_clip_grad
        if self.enable_clip_grad:
            print("Using grads clip.")

        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode not in ParallelMode.MODE_LIST:
            raise ValueError("Parallel mode does not support: ", self.parallel_mode)
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            print("Using DistributedGradReducer")
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = _get_device_num()
            self.grad_reducer = nn.DistributedGradReducer(self.optimizer.parameters, mean, degree)

    def construct(self, turns, every_turn_len, response, response_len, labels):
        """An iterative process"""
        turns = self.cast(turns, mstype.int32)
        every_turn_len = self.cast(every_turn_len, mstype.int32)
        response = self.cast(response, mstype.int32)
        response_len = self.cast(response_len, mstype.int32)
        labels = self.cast(labels, mstype.float32)

        weights = self.weights
        loss = self.network(turns, every_turn_len, response, response_len, labels)
        grads = self.grad(self.network, weights)(turns, every_turn_len, response, response_len, labels)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        if self.enable_clip_grad:  # grads clip
            grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        succ = self.optimizer(grads)

        return F.depend(loss, succ)


class PredictWithNet(nn.Cell):
    """To make predictions"""
    def __init__(self, network):
        super(PredictWithNet, self).__init__(auto_prefix=False)
        self.network = network
        self.cast = P.Cast()

    def construct(self, turns, every_turn_len, response, response_len, labels):
        """Process of prediction"""
        turns = self.cast(turns, mstype.int32)
        every_turn_len = self.cast(every_turn_len, mstype.int32)
        response = self.cast(response, mstype.int32)
        response_len = self.cast(response_len, mstype.int32)
        labels = self.cast(labels, mstype.float32)

        logits = self.network(turns, every_turn_len, response, response_len, labels)
        return logits, labels
