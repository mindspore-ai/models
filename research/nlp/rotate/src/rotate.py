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
""" RotatE Model """
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import Parameter, Tensor, ParameterTuple
from mindspore.nn.optim import Adam
from mindspore.ops import stop_gradient


np.random.seed(1)
VERY_SMALL_NUMBER = 1e-6


class MyNorm(nn.Cell):
    """
    Some data might be truncated under ms.dtype.float16 setting, thus generating 0 during backpropagation.
    We add a small bia to avoid generating dirty data.
    """

    def __init__(self, axis=0):
        super(MyNorm, self).__init__()
        self.axis = axis
        self.reduce_sum = P.ReduceSum(True)
        self.sqrt = P.Sqrt()
        self.squeeze = P.Squeeze(self.axis)

    def construct(self, x):
        """ Add bias before calculation sqrt """
        x = self.sqrt(self.reduce_sum(F.square(x), self.axis) + VERY_SMALL_NUMBER)
        x = self.squeeze(x)
        return x


class RotatE(nn.Cell):
    """ RotatE Model GRAPH_MODE """

    def __init__(self, config):
        super(RotatE, self).__init__()
        self.num_entity = config.num_entity
        self.num_relation = config.num_relation
        self.hidden_dim = config.hidden_dim
        self.epsilon = 2.0
        self.pi = 3.14159265358979323846
        self.gamma = config.gamma

        self.entity_dim = self.hidden_dim * 2 if config.double_entity_embedding else self.hidden_dim
        self.relation_dim = self.hidden_dim * 2 if config.double_relation_embedding else self.hidden_dim

        self.embedding_range = (config.gamma + self.epsilon) / self.hidden_dim  # 0.016
        self.entity_embedding = Parameter(
            Tensor(
                np.random.uniform(
                    -self.embedding_range,
                    self.embedding_range,
                    (self.num_entity, self.entity_dim)
                ), dtype=ms.dtype.float32
            )
        )
        self.relation_embedding = Parameter(
            Tensor(
                np.random.uniform(
                    -self.embedding_range,
                    self.embedding_range,
                    (self.num_relation, self.relation_dim)
                ), dtype=ms.dtype.float32
            )
        )

        self.expand_dim = P.ExpandDims()
        self.split = P.Split(axis=2, output_num=2)
        self.cos = P.Cos()
        self.sin = P.Sin()
        self.stack = P.Stack(axis=0)
        self.norm = MyNorm(axis=0)
        self.gather = P.Gather()

    def construct_head(self, sample):
        """ head negative sample mode """
        tail_part, head_part = sample
        batch_size, negative_sample_size = head_part.shape[0], head_part.shape[1]

        head = self.gather(self.entity_embedding, head_part.view(-1), 0)
        head = head.view(batch_size, negative_sample_size, -1)

        relation = self.gather(self.relation_embedding, tail_part[:, 1], 0)
        relation = self.expand_dim(relation, 1)

        tail = self.gather(self.entity_embedding, tail_part[:, 2], 0)
        tail = self.expand_dim(tail, 1)

        return self.score_head(head, relation, tail)

    def construct_tail(self, sample):
        """ positive negative sample mode """
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.shape[0], tail_part.shape[1]

        head = self.gather(self.entity_embedding, head_part[:, 0], 0)
        head = self.expand_dim(head, 1)

        relation = self.gather(self.relation_embedding, head_part[:, 1], 0)
        relation = self.expand_dim(relation, 1)

        tail = self.gather(self.entity_embedding, tail_part.view(-1), 0)
        tail = tail.view(batch_size, negative_sample_size, -1)

        return self.score_tail_single(head, relation, tail)

    def construct_single(self, sample):
        """ positive sample mode """
        head = self.gather(self.entity_embedding, sample[:, 0], 0)
        head = self.expand_dim(head, 1)

        relation = self.gather(self.relation_embedding, sample[:, 1], 0)
        relation = self.expand_dim(relation, 1)

        tail = self.gather(self.entity_embedding, sample[:, 2], 0)
        tail = self.expand_dim(tail, 1)

        return self.score_tail_single(head, relation, tail)

    def score_head(self, head, relation, tail):
        """
        score function of head-mode data

        Args:
            head: (Tensor) (batch_size, negative_sample_size, entity_dim)
            relation: (Tensor) (batch_size, 1, relation_dim)
            tail: (Tensor) (batch_size, 1, entity_dim)

        Returns:
            score(Tensor): (batch_size, negative_sample_size)

        """
        re_head, im_head = self.split(head)
        re_tail, im_tail = self.split(tail)

        phase_relation = relation / (self.embedding_range / self.pi)

        re_relation = self.cos(phase_relation)
        im_relation = self.sin(phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head

        score = self.stack([re_score, im_score])
        score = self.norm(score)

        score = self.gamma - score.sum(axis=2)
        return score

    def score_tail_single(self, head, relation, tail):
        """
        score function of tail-mode data and positive sample data

        Args:
            head (Tensor): (batch_size, 1, entity_dim)
            relation(Tensor): (batch_size, 1, relation_dim)
            tail (Tensor): (batch_size, negative_sample_size, entity_dim)

        Returns:
            score(Tensor): (batch_size, negative_sample_size)

        """
        re_head, im_head = self.split(head)
        re_tail, im_tail = self.split(tail)

        phase_relation = relation / (self.embedding_range / self.pi)

        re_relation = self.cos(phase_relation)
        im_relation = self.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = self.stack([re_score, im_score])
        score = self.norm(score)

        score = self.gamma - score.sum(axis=2)

        return score


class NetWithLossClass(nn.Cell):
    """
    Calculate self-adversarial negative sampling loss
    """

    def __init__(self, network, config, mode='head-mode'):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.network = network
        self.adversarial_temperature = config.adversarial_temperature
        self.mode = mode

        self.softmax_axis_1 = P.Softmax(axis=1)
        self.squeeze_axis_1 = P.Squeeze(axis=1)
        self.logsigmoid = nn.LogSigmoid()

    def construct(self, positive_sample, negative_sample, subsampling_weight):
        """
        Calculate self-adversarial negative sampling loss

        Args:
            positive_sample (Tensor): (batch_size, )
            negative_sample (Tensor): (batch_size, )
            subsampling_weight (Tensor): (batch_size, 1)

        Returns:
            loss(Tensor)

        """
        subsampling_weight = self.squeeze_axis_1(subsampling_weight)
        if self.mode == 'head-mode':
            negative_score = self.network.construct_head((positive_sample, negative_sample))
        else:
            negative_score = self.network.construct_tail((positive_sample, negative_sample))

        negative_score = (stop_gradient(self.softmax_axis_1(negative_score * self.adversarial_temperature))
                          * self.logsigmoid(-negative_score)).sum(axis=1)

        positive_score = self.network.construct_single(positive_sample)
        positive_score = self.squeeze_axis_1(self.logsigmoid(positive_score))

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        return loss


def get_lr_rotate(current_step, lr_max, max_step):
    """
    Warmup training strategy.

    Args:
        current_step (int): current step
        lr_max (float): original learning rate
        max_step (int): total steps

    Returns:
        list of learning rate at each step

    """
    lr_each_step = []
    decay_epoch_index = [0.5 * max_step]
    for i in range(max_step):
        if i < decay_epoch_index[0]:
            lr = lr_max
        else:
            lr = lr_max * 0.1
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


class ModelBuilder:
    """
    Model builder for RotatE.

    Args:
        model_config (ModelConfig): Model configuration.
    """
    def __init__(self, model_config):
        self.model_config = model_config

    def get_train_net(self):
        """ Get Networks for Training. """
        kge_net = RotatE(config=self.model_config)
        loss_net_head = NetWithLossClass(network=kge_net, config=self.model_config, mode='head-mode')
        loss_net_tail = NetWithLossClass(network=kge_net, config=self.model_config, mode='tail-mode')

        if self.model_config.use_dynamic_loss_scale:
            manager = nn.DynamicLossScaleUpdateCell(
                loss_scale_value=2 ** 10,
                scale_factor=2,
                scale_window=2000
            )
        else:
            manager = nn.FixedLossScaleUpdateCell(
                loss_scale_value=2 ** 10
            )
        lr = get_lr_rotate(
            current_step=0,
            lr_max=self.model_config.lr,
            max_step=self.model_config.max_steps
        )
        weights = ParameterTuple(loss_net_head.trainable_params())
        optimizer = Adam(weights, learning_rate=Tensor(lr, dtype=ms.dtype.float32))
        train_net_head = nn.TrainOneStepWithLossScaleCell(
            network=loss_net_head,
            optimizer=optimizer,
            scale_sense=manager
        )
        train_net_tail = nn.TrainOneStepWithLossScaleCell(
            network=loss_net_tail,
            optimizer=optimizer,
            scale_sense=manager
        )
        return train_net_head, train_net_tail

    def get_eval_net(self):
        """ Get Networks for Evaluation. """
        eval_net = RotatE(config=self.model_config)
        return eval_net
