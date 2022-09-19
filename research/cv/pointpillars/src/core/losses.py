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
"""losses"""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.ops import constexpr


@constexpr
def _create_off_value():
    """create off value"""
    return Tensor(0.0, mstype.float32)


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0):
    """Creates dense vector with indices set to specific value and rest to zeros.

    This function exists because it is unclear if it is safe to use
      tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

    Args:
      indices: 1d Tensor with integer indices which are to be set to
        indices_values.
      size: scalar with size (integer) of output Tensor.
      indices_value: values of elements specified by indices in the output vector
      default_value: values of other elements in the output vector.

    Returns:
      dense 1D Tensor of shape [size] with indices set to indices_values and the
      rest set to default_value.
    """
    dense = ops.Zeros()(size).fill(default_value)
    dense[indices] = indices_value

    return dense


def _sigmoid_cross_entropy_with_logits(logits, labels):
    """sigmoid cross entropy with logits"""
    loss = ops.clip_by_value(
        logits,
        clip_value_min=_create_off_value(),
        clip_value_max=logits.max()
    ) - logits * labels.astype(logits.dtype)
    loss += ops.Log1p()(ops.Exp()(-ops.Abs()(logits)))
    return loss


def _softmax_cross_entropy_with_logits(logits, labels):
    """softmax cross entropy with logits"""
    loss_ftor = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    loss = loss_ftor(logits.astype(mstype.float32), ops.ArgMaxWithValue(axis=-1)(labels)[0].astype(mstype.int32))
    return loss


class SigmoidFocalClassificationLoss(nn.Cell):
    """Sigmoid focal cross entropy loss.

    Focal loss down-weights well classified examples and focuses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def construct(self,
                  prediction_tensor,
                  target_tensor,
                  weights,
                  class_indices=None):
        """Compute loss function."""
        weights = ops.ExpandDims()(weights, 2)
        if class_indices is not None:
            weights *= indices_to_dense_vector(
                class_indices,
                prediction_tensor.shape[2]
            ).view(1, 1, -1).astype(prediction_tensor.dtype)
        per_entry_cross_ent = (
            _sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor)
        )
        prediction_probabilities = ops.Sigmoid()(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = ops.Pow()(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha +
                                   (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
        return focal_cross_entropy_loss * weights


class WeightedSmoothL1LocalizationLoss(nn.Cell):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, code_weights=None, codewise=True):
        super().__init__()
        self._sigma = sigma
        if code_weights is not None:
            self._code_weights = np.array(code_weights, dtype=np.float32)
            self._code_weights = Tensor(self._code_weights)
        else:
            self._code_weights = None
        self._codewise = codewise

    def construct(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              code_size] representing the (encoded) predicted locations of objects.
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              code_size] representing the regression targets
            weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
            loss: a float tensor of shape [batch_size, num_anchors] tensor
                representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            code_weights = self._code_weights.astype(prediction_tensor.dtype)
            diff = code_weights.view(1, 1, -1) * diff
        abs_diff = ops.Abs()(diff)
        abs_diff_lt_1 = ops.LessEqual()(abs_diff, 1 / (self._sigma ** 2)).astype(abs_diff.dtype)
        loss = (abs_diff_lt_1 * 0.5 * ops.Pow()(abs_diff * self._sigma, 2)
                + (abs_diff - 1 / (2 * (self._sigma ** 2))) * (1. - abs_diff_lt_1))
        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= ops.ExpandDims()(weights, -1)
        else:
            anchorwise_smooth_l1norm = ops.ReduceSum()(loss, 2)
            if weights is not None:
                anchorwise_smooth_l1norm *= weights
        return anchorwise_smooth_l1norm


class WeightedSoftmaxClassificationLoss(nn.Cell):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0):
        """Constructor.

        Args:
          logit_scale: When this value is high, the prediction is "diffused" and
                       when this value is low, the prediction is made peakier.
                       (default 1.0)
        """
        super().__init__()
        self._logit_scale = logit_scale

    def construct(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors]
            representing the value of the loss function.
        """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = ops.Div()(prediction_tensor, self._logit_scale)
        per_row_cross_ent = _softmax_cross_entropy_with_logits(
            labels=target_tensor.view(-1, num_classes),
            logits=prediction_tensor.view(-1, num_classes)
        )
        return per_row_cross_ent.view(weights.shape) * weights
