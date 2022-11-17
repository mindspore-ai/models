# Copyright 2020 Huawei Technologies Co., Ltd
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

'''
Bert for finetune script.
'''

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from .bert_for_pre_training import clip_grad
from .finetune_eval_model import BertCLSModel, BertNERModel, BertSquadModel
from .utils import CrossEntropyCalculation

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class BertFinetuneCell(nn.TrainOneStepWithLossScaleCell):
    """
    Especially defined for finetuning where only four inputs tensor are needed.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Different from the builtin loss_scale wrapper cell, we apply grad_clip before the optimization.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertFinetuneCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  sens=None):
        """Bert Finetune"""

        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        if sens is None:
            scaling_sens = self.scale_sense
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


class BertSquadCell(nn.TrainOneStepWithLossScaleCell):
    """
    specifically defined for finetuning where only four inputs tensor are needed.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertSquadCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  start_position,
                  end_position,
                  unique_id,
                  is_impossible,
                  sens=None):
        """BertSquad"""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            start_position,
                            end_position,
                            unique_id,
                            is_impossible)
        if sens is None:
            scaling_sens = self.scale_sense
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 start_position,
                                                 end_position,
                                                 unique_id,
                                                 is_impossible,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


class BertCLS(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLS, self).__init__()
        self.bert = BertCLSModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,
                                 assessment_method)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.is_training = is_training

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.assessment_method == "spearman_correlation":
            if self.is_training:
                loss = self.loss(logits, label_ids)
            else:
                loss = logits
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss


class BertNER(nn.Cell):
    """
    Train interface for sequence labeling finetuning task.
    """

    def __init__(self, config, batch_size, is_training, num_labels=11, use_crf=False, with_lstm=False,
                 tag_to_index=None, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertNER, self).__init__()
        self.bert = BertNERModel(config, is_training, num_labels, use_crf, with_lstm, dropout_prob,
                                 use_one_hot_embeddings)
        if use_crf:
            if not tag_to_index:
                raise Exception("The dict for tag-index mapping should be provided for CRF.")
            from src.CRF import CRF
            self.loss = CRF(tag_to_index, batch_size, config.seq_length, is_training)
        else:
            self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.use_crf = use_crf

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.use_crf:
            loss = self.loss(logits, label_ids)
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss


class BertSquad(nn.Cell):
    '''
    Train interface for SQuAD finetuning task.
    '''

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSquad, self).__init__()
        self.bert = BertSquadModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.seq_length = config.seq_length
        self.is_training = is_training
        self.total_num = Parameter(Tensor([0], mstype.float32))
        self.start_num = Parameter(Tensor([0], mstype.float32))
        self.end_num = Parameter(Tensor([0], mstype.float32))
        self.sum = P.ReduceSum()
        self.equal = P.Equal()
        self.argmax = P.ArgMaxWithValue(axis=1)
        self.squeeze = P.Squeeze(axis=-1)

    def construct(self, input_ids, input_mask, token_type_id, start_position, end_position, unique_id, is_impossible):
        """interface for SQuAD finetuning task"""
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.is_training:
            unstacked_logits_0 = self.squeeze(logits[:, :, 0:1])
            unstacked_logits_1 = self.squeeze(logits[:, :, 1:2])
            start_loss = self.loss(unstacked_logits_0, start_position, self.seq_length)
            end_loss = self.loss(unstacked_logits_1, end_position, self.seq_length)
            total_loss = (start_loss + end_loss) / 2.0
        else:
            start_logits = self.squeeze(logits[:, :, 0:1])
            start_logits = start_logits + 100 * input_mask
            end_logits = self.squeeze(logits[:, :, 1:2])
            end_logits = end_logits + 100 * input_mask
            total_loss = (unique_id, start_logits, end_logits)
        return total_loss
