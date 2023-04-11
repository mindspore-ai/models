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
"""
Loss function definition.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import nn, ops, Tensor, Parameter, context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import _get_gradients_mean

from nnmnkwii import preprocessing as P1
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw
from wavenet_vocoder.mixture import discretized_mix_logistic_loss
from wavenet_vocoder.mixture import mix_gaussian_loss
from train_pytorch import to_categorical
from tqdm import tqdm
import audio

matplotlib.use('Agg')


def sequence_mask(sequence_length, max_len=None):
    """make sequence mask"""
    sequence_length = sequence_length.asnumpy()
    if max_len is None:
        max_len = np.max(sequence_length)
    batch_size = sequence_length.shape[0]
    seq_range = np.linspace(0, max_len - 1, max_len, dtype=np.int32)
    seq_range_expand = np.tile(np.expand_dims(seq_range, 0), (batch_size, 1))
    seq_length_expand = np.tile(np.expand_dims(sequence_length, 1), (1, max_len))
    seq_length_expand = np.expand_dims(np.array(seq_range_expand < seq_length_expand, dtype=np.float32), -1)
    return Tensor(seq_length_expand)


class MaskedCrossEntropyLoss(nn.Cell):
    """MaskedCrossEntropyLoss"""

    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, inputs, target):
        losses = self.criterion(inputs, target)
        return losses


class DiscretizedMixturelogisticLoss(nn.Cell):
    """DiscretizedMixturelogisticLoss"""

    def __init__(self, hparams):
        super(DiscretizedMixturelogisticLoss, self).__init__()
        self.quantize_channels = hparams.quantize_channels
        self.log_scale_min = hparams.log_scale_min
        self.discretized_mix_logistic_loss = discretized_mix_logistic_loss(num_classes=hparams.quantize_channels,
                                                                           log_scale_min=hparams.log_scale_min,
                                                                           reduce=False)
        self.reduce_sum_op = P.ReduceSum()
        self.reduce_mean_op = P.ReduceMean()

    def construct(self, inputs, target, mask=None):
        losses = self.discretized_mix_logistic_loss(inputs, target)
        return self.reduce_sum_op(losses * mask) / self.reduce_sum_op(mask)


class MixtureGaussianLoss(nn.Cell):
    """MixtureGaussianLoss"""

    def __init__(self, hparams):
        super(MixtureGaussianLoss, self).__init__()
        self.quantize_channels = hparams.quantize_channels
        self.log_scale_min = hparams.log_scale_min
        self.mix_gaussian_loss = mix_gaussian_loss(log_scale_min=hparams.log_scale_min, reduce=False)
        self.reduce_sum_op = P.ReduceSum()
        self.reduce_mean_op = P.ReduceMean()

    def construct(self, inputs, target, mask=None):
        """

        Args:
            inputs (Tensor): Predicted distribution
            target (Tensor): Target
            mask (Tensor): Mask

        Returns:
            Tensor: Loss tensor

        """
        losses = self.mix_gaussian_loss(inputs, target)
        return self.reduce_sum_op(losses * mask) / self.reduce_sum_op(mask)


def save_waveplot(path, y_hat, y_target, sample_rate):
    sr = sample_rate
    plt.figure(figsize=(16, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(y_target, sr=sr)
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(y_hat, sr=sr)
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def eval_model(hparams, global_step, model, x, y, c, g, input_lengths, eval_dir):
    """
    Function for model evaluation. This function is used for debugging in this project.
    """

    model.set_train(False)
    idx = np.random.randint(0, len(y))
    length = input_lengths.asnumpy()[idx]
    y_target = np.reshape(y.asnumpy()[idx], (-1))
    y_target = y_target[:length]

    if c is not None:
        expand_op = P.ExpandDims()
        if hparams.upsample_conditional_features:
            c = expand_op(c[idx, :, :int(length // audio.get_hop_size() + hparams.cin_pad * 2)], 0)
        else:
            c = expand_op(c[idx, :, :length], 0)
        assert c.dim() == 3
        print("Shape of local conditioning features: {}".format(c.size()))

    if g is not None:
        g = g[idx]
        print("Shape of global conditioning features: {}".format(g.size()))

    # Dummy silence
    if is_mulaw_quantize(hparams.input_type):
        initial_value = P1.mulaw_quantize(0, hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        initial_value = P1.mulaw(0.0, hparams.quantize_channels)
    else:
        initial_value = 0.0

    if is_mulaw_quantize(hparams.input_type):
        initial_input = to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = Tensor(np.reshape(initial_input, (1, 1, hparams.quantize_channels)))

    else:
        initial_input = np.ones((1, 1, 1)) * initial_value
        initial_input = Tensor(initial_input)

    # Run the model in fast eval mode
    y_hat = model.incremental_forward(initial_input, c=c, g=g, T=length, softmax=True, quantize=True, tqdm=tqdm,
                                      log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = np.reshape(np.argmax(y_hat, 1), (-1))
        y_hat = P1.inv_mulaw_quantize(y_hat, hparams.quantize_channels - 1)
        y_target = P1.inv_mulaw_quantize(y_target, hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        y_hat = P1.inv_mulaw(np.reshape(y_hat, (-1)), hparams.quantize_channels)
        y_target = P1.inv_mulaw(y_target, hparams.quantize_channels)
    else:
        y_hat = np.reshape(y_hat, (-1))

    # Save audio
    os.makedirs(eval_dir, exist_ok=True)
    path = os.path.join(eval_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)

    path = os.path.join(eval_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y_target, sr=hparams.sample_rate)

    # Save figure
    path = os.path.join(eval_dir, "step{:09d}_waveplots.png".format(global_step))
    save_waveplot(path, y_hat, y_target, hparams.sample_rate)


class PredictNet(nn.Cell):
    """
    NetWithLossClass definition
    """

    def __init__(self, network):
        super(PredictNet, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, x, c, g):
        y_hat = self.network(x, c, g, False)
        return y_hat


class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition

    Args:
        network (Cell): Pre-defined WaveNet.
        hparams (optional): Parameters.

    Returns:
        Tensor, loss tensor.
    """

    def __init__(self, network, hparams):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.network = network
        self.hparams = hparams
        self.ReduceMean_false = P.ReduceMean(keep_dims=False)
        self.expand_op = P.ExpandDims()
        self.transpose_op = P.Transpose()
        self.reshape_op = P.Reshape()
        self.is_mulaw_quant = is_mulaw_quantize(hparams.input_type)
        if self.is_mulaw_quant:
            self.criterion = MaskedCrossEntropyLoss()
        else:
            if hparams.output_distribution == "Logistic":
                self.criterion = DiscretizedMixturelogisticLoss(hparams)
            elif hparams.output_distribution == "Normal":
                self.criterion = MixtureGaussianLoss(hparams)
            else:
                self.criterion = None
                raise RuntimeError(
                    "Not supported output distribution type: {}".format(hparams.output_distribution))

    def construct(self, x, y, c, g, input_lengths, mask):
        """

        Args:
            x (Tensor): input.
            y (Tensor): prediction.
            c (Tensor): Local conditioning feature.
            g (Tensor): Global conditioning feature.
            input_lengths(Tensor): input_lengths.
            mask (Tensor): Padding mask.

        Returns:
            Tensor: Loss tensor

        """
        y_hat = self.network(x, c, g, False)
        if self.is_mulaw_quant:
            y_hat = self.transpose_op(y_hat[:, :, :-1], (0, 2, 1))
            y_hat = self.reshape_op(y_hat, (-1, y_hat.shape[-1]))
            y = self.reshape_op(y[:, 1:, 0], (-1,))
            loss = self.criterion(y_hat, y)
        else:
            loss = self.criterion(y_hat[:, :, :-1], y[:, 1:, :], mask[:, 1:, :])
        return loss


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


compute_norm = C.MultitypeFuncGraph("compute_norm")


@compute_norm.register("Tensor")
def _compute_norm(grad):
    norm = ops.norm(F.cast(grad, ms.float32))
    ret = F.expand_dims(F.cast(norm, ms.float32), 0)
    return ret


grad_div = C.MultitypeFuncGraph("grad_div")


@grad_div.register("Tensor", "Tensor")
def _grad_div(val, grad):
    div = P.RealDiv()
    mul = P.Mul()
    scale = div(1.0, val)
    ret = mul(grad, scale)
    return ret


class WaveNetTrainOneStepWithLossScaleCell(nn.Cell):
    """
    WaveNet training with loss scaling.

    Args:
        network (Cell): The training WaveNet.
        optimizer (Cell): Optimizer for updating the weights.
        scale_sense (Cell): The loss scaling update logic cell.

    Returns:
        Tuple[Tensor, Tensor, Tensor], loss, overflow, sen.
    """

    def __init__(self, network, optimizer, scale_update_cell):
        super(WaveNetTrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.add_flags(has_effect=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer

        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = 1.0
        self.fill = P.Fill()
        self.dtype = P.DType()
        self.get_shape = P.Shape()
        self.cast = P.Cast()
        self.concat = P.Concat()
        self.less_equal = P.LessEqual()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.scalar_summary = P.ScalarSummary()
        self.greater = P.Greater()
        self.select = P.Select()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.is_distributed = False
        self.base = Tensor(1, ms.float32)

        self.all_reduce = P.AllReduce()

        self.loss_scaling_manager = scale_update_cell
        self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ms.float32))

        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
            self.is_distributed = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            mean = _get_gradients_mean()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, self.degree)

    def construct(self, x, y, c, g, input_lengths, mask):
        """

        Args:
            x (Tensor): Source audio signal.
            y (Tensor): Target audio signal.
            c (Tensor): Local conditioning feature.
            g (Tensor): Global conditioning feature.
            input_lengths(Tensor): input_lengths
            mask (Tensor): Padding mask.

        Returns:
            Tuple[Tensor, Tensor, Tensor], loss, overflow, sen.

        """
        weights = self.weights
        loss = self.network(x, y, c, g, input_lengths, mask)

        scale_sense = self.loss_scale
        # Alloc status.
        init = self.alloc_status()
        init = F.depend(init, loss)

        # Clear overflow buffer.
        clear_status = self.clear_before_grad(init)
        scale_sense = F.depend(scale_sense, clear_status)

        grads = self.grad(self.network, weights)(x, y, c, g, input_lengths, mask, self.cast(scale_sense, ms.float32))
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, self.degree * scale_sense), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))

        if self.is_distributed:
            # Sum overflow flag over devices.
            flag_reduce = self.all_reduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)

        overflow = self.loss_scaling_manager(self.loss_scale, cond)

        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)

        self.scalar_summary("training.loss", loss)

        ret = (loss, scale_sense.value())
        return F.depend(ret, succ)
