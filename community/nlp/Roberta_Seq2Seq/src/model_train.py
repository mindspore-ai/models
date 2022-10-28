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

"""model for train"""

from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import ops
from src.utils import MaskedFill


class LabelSmoothedCrossEntropyCriterion(nn.Cell):
    """
    Label Smoothed Cross-Entropy Criterion.

    Args:
        config (TransformerConfig): The config of Transformer.

    Returns:
        Tensor, final loss.
    """

    def __init__(self, config):
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
        self.vocab_size = config.vocab_size
        self.onehot = P.OneHot()
        self.on_value = Tensor(
            float(1 - config.label_smoothing), mstype.float32)
        self.off_value = Tensor(
            config.label_smoothing / float(self.vocab_size - 1), mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.flatten = P.Flatten()
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.flat_shape = (config.batch_size * config.seq_length,)
        self.get_shape = P.Shape()
        self.expand_dims = P.ExpandDims()

    def construct(self, prediction_scores, label_ids, label_weights):
        """
        Construct network to calculate loss.

        Args:
            prediction_scores (Tensor): Prediction scores.
            label_ids (Tensor): Labels.
            label_weights (Tensor): Mask tensor.

        Returns:
            Tensor, final loss.
        """
        label_shape = self.get_shape(label_ids)

        label_ids = self.reshape(label_ids, (label_shape[0] * label_shape[1],))
        label_weights = self.cast(
            self.reshape(label_weights, (label_shape[0] * label_shape[1],)),
            mstype.float32
        )
        one_hot_labels = self.onehot(
            label_ids, self.vocab_size, self.on_value, self.off_value)
        prediction_shape = self.get_shape(prediction_scores)
        prediction_scores = self.reshape(
            prediction_scores, (prediction_shape[2], prediction_shape[0] * prediction_shape[1]))
        per_example_loss = self.neg(self.reduce_sum(
            prediction_scores * one_hot_labels, self.last_idx))
        numerator = self.reduce_sum(label_weights * per_example_loss, ())
        denominator = self.reduce_sum(
            label_weights, ()) + self.cast(F.tuple_to_array((1e-5,)), mstype.float32)
        loss = numerator / denominator

        return loss


class LabelSmoothedNllLoss(nn.Cell):
    """
    Args:
        lprobs:
        target:
        epsilon:
        ignore_index:
    Returns:
    """
    def __init__(self):
        super(LabelSmoothedNllLoss, self).__init__()
        self.expand_dims = ops.ExpandDims()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.squeeze = ops.Squeeze(-1)
        self.equal = ops.Equal()
        self.masked_fill = MaskedFill()
        self.cast = ops.Cast()

    def construct(self, lprobs, target, epsilon, ignore_index=-100):
        """

        Args:
            lprobs:
            target:
            epsilon:
            ignore_index:

        Returns:

        """
        if target.ndim == lprobs.ndim - 1:
            target = self.expand_dims(target, -1)
        lprobs.astype(mstype.float32)

        nll_loss = ops.GatherD()(-lprobs, -1, target)
        smooth_loss = -lprobs.sum(axis=-1, keepdims=True)
        nll_loss = self.cast(nll_loss, mstype.float32)
        smooth_loss = self.cast(smooth_loss, mstype.float32)

        if ignore_index is not None:
            pad_mask = self.equal(target, ignore_index)
            nll_loss = self.masked_fill(nll_loss, pad_mask, 0.0)
            smooth_loss = self.masked_fill(smooth_loss, pad_mask, 0.0)
        else:
            nll_loss = self.squeeze(nll_loss)
            smooth_loss = self.squeeze(smooth_loss)

        nll_loss = nll_loss.mean()  # mean()? Scared to break other math. sum()
        smooth_loss = smooth_loss.mean()
        eps_i = epsilon / lprobs.shape[-1]
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss


class EncoderDecoderWithLossCell(nn.Cell):
    """
    Args:
        backbone:
        loss_fn:
        pad_token_id:
        label_smoothing:
        data_args:
    """
    def __init__(self, backbone, loss_fn, pad_token_id, label_smoothing, data_args=None):
        super(EncoderDecoderWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        self.data_args = data_args
        self.ignore_pad_token_for_loss = None
        if data_args is not None:
            self.ignore_pad_token_for_loss = data_args.ignore_pad_token_for_loss
        else:
            self.ignore_pad_token_for_loss = False
        self.log_softmax = nn.LogSoftmax()

    def construct(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        """

        Args:
            input_ids:
            attention_mask:
            decoder_input_ids:
            decoder_attention_mask:
            labels:

        Returns:

        """
        if self.label_smoothing == 0.0:
            if self.ignore_pad_token_for_loss:
                logits = self._backbone(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=decoder_attention_mask,
                                        use_cache=False)[0]
                loss = self._loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:
                loss, logits = self._backbone(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              decoder_input_ids=decoder_input_ids,
                                              decoder_attention_mask=decoder_attention_mask,
                                              labels=labels,
                                              use_cache=False)[:2]
        else:
            logits = self._backbone(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask,
                                    use_cache=False)[0]
            lprobs = self.log_softmax(logits)
            loss = self._loss_fn(
                lprobs, labels, self.label_smoothing)
        return loss
