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


import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common import initializer as ini
from mindspore.boost import GradientAccumulation
from mindspore.ops import functional as F
from mindspore.common import RowTensor
import mindspore.common.dtype as mstype
import mindspore.numpy as np
import numpy as np

activations = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
}

TRAIN_INPUT_PAD_LENGTH = 1500
TRAIN_LABEL_PAD_LENGTH = 350
TEST_INPUT_PAD_LENGTH = 3500

_grad_scale = P.composite.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale),
                                          F.dtype(grad.values)),
                     grad.dense_shape)


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


class MaskedConv1d(nn.Cell):
    """1D convolution with sequence masking
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, group=1,
                 padding=0, pad_mode='pad', weight_init='xavier_uniform',
                 has_bias=False, masked=True):
        super(MaskedConv1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, pad_mode=pad_mode, weight_init=weight_init,
                               dilation=dilation, group=group, has_bias=has_bias)

        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.masked = masked
        self.indices_max = mindspore.Tensor(np.arange(TRAIN_INPUT_PAD_LENGTH))
        self.expand_dims = mindspore.ops.ExpandDims()
        self.unsqueeze = mindspore.ops.ExpandDims()

    def get_seq_len(self, lens):
        return ((lens + 2 * self.padding - self.dilation
                 * (self.kernel_size - 1) - 1) // self.stride + 1)

    def construct(self, x, x_lens):

        if self.masked:
            x_shape = P.shape(x)
            max_length = x_shape[2]
            indices = self.indices_max[:max_length]
            indices = self.expand_dims(indices, 0)
            mask = indices < self.unsqueeze(x_lens, 1)
            x = x * self.unsqueeze(mask, 1)
            x_lens = self.get_seq_len(x_lens)
            x = self.conv1(x)

        return x, x_lens


class Mbatchnorm(nn.Cell):
    def __init__(self, num_features, eps, momentum):
        super(Mbatchnorm, self).__init__()

        self.batchnorm = nn.BatchNorm2d(
            num_features=num_features, eps=eps, momentum=momentum)

    def construct(self, x):
        shape = P.shape(x)
        x = x.reshape((shape[0], shape[1], shape[2], -1))
        x = self.batchnorm(x)
        out = x.reshape((shape[0], shape[1], shape[2]))

        return out


class JasperBlock(nn.Cell):
    __constants__ = ["use_conv_masks"]

    """Jasper Block. See https://arxiv.org/pdf/1904.03288.pdf
    """

    def __init__(self, infilters, filters, repeat, kernel_size, stride,
                 dilation, pad_mode='pad', dropout=0.2, activation=None,
                 residual=True, residual_panes=None, use_conv_masks=False):
        super(JasperBlock, self).__init__()
        if residual_panes is None:
            residual_panes = []
        padding_val = get_same_padding(kernel_size, stride, dilation)
        self.use_conv_masks = use_conv_masks
        self.conv = nn.CellList()
        for i in range(repeat):
            self.conv.extend(self._conv_bn(infilters if i == 0 else filters,
                                           filters,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           dilation=dilation,
                                           padding=padding_val,
                                           pad_mode=pad_mode))

            if i < repeat - 1:
                self.conv.extend(self._act_dropout(dropout, activation))

        self.res = nn.CellList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            if residual_panes is None:
                res_panes = [infilters]
                self.dense_residual = False
            for ip in res_panes:
                self.res.append(nn.CellList(
                    self._conv_bn(ip, filters, kernel_size=1)))
        self.out = nn.SequentialCell(*self._act_dropout(dropout, activation))

    def _conv_bn(self, in_channels, out_channels, **kw):
        return [MaskedConv1d(in_channels, out_channels,
                             masked=self.use_conv_masks, **kw),
                Mbatchnorm(num_features=out_channels, eps=1e-3, momentum=0.9)]

    def _act_dropout(self, dropout=0.2, activation=None):
        return [activation,
                nn.Dropout(p=dropout)]

    def construct(self, xs, xs_lens=None):
        if not self.use_conv_masks:
            xs_lens = 0
        out = xs[-1]
        lens = xs_lens
        for i, l in enumerate(self.conv):
            if i % 4 == 0:
                out, lens = l(out, lens)
            else:
                out = l(out)
        # residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if j == 0:
                        res_out, lens = res_layer(res_out, xs_lens)
                    else:
                        res_out = res_layer(res_out)
                out += res_out
        # output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]
        if self.use_conv_masks:
            return out, lens
        return out, None


class JasperEncoder(nn.Cell):
    __constants__ = ["use_conv_masks"]

    def __init__(self, in_feats=10, frame_splicing=1, activation="relu",
                 weight_init='xavier_uniform', use_conv_masks=False, blocks=None):
        super(JasperEncoder, self).__init__()
        if blocks is None:
            blocks = []
        self.use_conv_masks = use_conv_masks
        self.layers = nn.CellList()
        in_feats *= frame_splicing
        all_residual_panes = []
        for _, blk in enumerate(blocks):
            blk['activation'] = activations[activation]()
            has_residual_dense = blk.pop('residual_dense', False)
            if has_residual_dense:
                all_residual_panes += [in_feats]
                blk['residual_panes'] = all_residual_panes
            else:
                blk['residual_panes'] = []
            self.layers.append(
                JasperBlock(infilters=in_feats, use_conv_masks=use_conv_masks, **blk))
            in_feats = blk['filters']

    def construct(self, x, x_lens=None):
        out, out_lens = [x], x_lens
        for l in self.layers:
            out, out_lens = l(out, out_lens)
        return out, out_lens


class JasperDecoderForCTC(nn.Cell):
    def __init__(self, in_feats=10, n_classes=3, init='xavier_uniform'):
        super(JasperDecoderForCTC, self).__init__()

        self.layers = nn.Conv1d(in_channels=in_feats, out_channels=n_classes,
                                kernel_size=1, has_bias=True, weight_init='xavier_uniform')

        self.transpose = P.Transpose()
        self.logsoftmax = nn.LogSoftmax()

    def construct(self, enc_out):
        out = self.layers(enc_out[-1])
        out = self.transpose(out, (0, 2, 1))
        out_2d = mindspore.ops.reshape(out, (-1, out.shape[2]))
        out_2d = self.logsoftmax(out_2d)
        out = self.transpose(mindspore.ops.reshape(
            out_2d, out.shape), (1, 0, 2))
        return out


class GreedyCTCDecoder(nn.Cell):

    def __init__(self):
        super().__init__()
        self.cast = mindspore.ops.Cast()
        self.fill = mindspore.ops.Fill()
        self.select = mindspore.ops.Select()
        self.argmax = np.argmax()

    def construct(self, log_probs, log_prob_lens=None):

        if log_prob_lens is not None:
            max_len = log_probs.size(1)
            idxs = np.arange(max_len, dtype=log_prob_lens.dtype)
            idxs = mindspore.Tensor(idxs)
            mask = np.expand_dims(idxs, 0) >= np.expand_dims(log_prob_lens, 1)
            mask = self.cast(mask, mstype.bool_)
            masked_value = self.fill(mindspore.float16, log_probs.shape, 0)
            log_probs = self.select(mask, masked_value, log_probs)
            out = self.argmax(log_probs, axis=-1)
        return out.astype("int")


class Jasper(nn.Cell):
    def __init__(self, encoder_kw=None, decoder_kw=None):
        super(Jasper, self).__init__()
        if encoder_kw is None:
            encoder_kw = {}
        if decoder_kw is None:
            decoder_kw = {}
        self.encoder = JasperEncoder(**encoder_kw)
        self.decoder = JasperDecoderForCTC(**decoder_kw)

    def construct(self, x, x_lens=None):
        enc, enc_lens = self.encoder(x, x_lens)
        out = self.decoder(enc)
        return out, enc_lens


class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition
    """

    def __init__(self, network, ascend=False):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        if ascend:
            self.loss = P.CTCLoss(ctc_merge_repeated=True,
                                  ignore_longer_outputs_than_inputs=True)
        else:
            self.loss = P.CTCLoss(ctc_merge_repeated=True)
        self.network = network
        self.ReduceMean_false = P.ReduceMean(keep_dims=False)
        self.squeeze_op = P.Squeeze(0)
        self.cast_op = P.Cast()

    def construct(self, inputs, input_length, target_indices, label_values):
        predict, output_length = self.network(inputs, input_length)
        predict = self.cast_op(predict, mstype.float32)
        loss = self.loss(predict, target_indices, label_values,
                         self.cast_op(output_length, mstype.int32))
        return self.ReduceMean_false(loss[0])


class TrainGradAccumulationStepsCell(nn.TrainOneStepWithLossScaleCell):
    """construct train accu step cell"""

    def __init__(self, network, optimizer, scale_sense, max_accumulation_step=2):
        super(TrainGradAccumulationStepsCell, self).__init__(
            network, optimizer, scale_sense)
        self.max_accumulation_step = max_accumulation_step
        self.grad_accumulation = GradientAccumulation(
            self.max_accumulation_step, self.optimizer)

    def construct(self, *inputs):

        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = P.composite.ones_like(
            loss) * P.functional.cast(scaling_sens, P.functional.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(P.functional.partial(
            _grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            loss = self.grad_accumulation(loss, grads)
        return loss


class PredictWithSoftmax(nn.Cell):
    """
    PredictWithSoftmax
    """

    def __init__(self, network):
        super(PredictWithSoftmax, self).__init__(auto_prefix=False)
        self.network = network
        self.inference_softmax = P.Softmax(axis=-1)
        self.transpose_op = P.Transpose()
        self.cast_op = P.Cast()

    def construct(self, inputs, input_length):
        x, output_sizes = self.network(
            inputs, self.cast_op(input_length, mstype.int32))
        x = self.inference_softmax(x)
        x = self.transpose_op(x, (1, 0, 2))
        return x, output_sizes


def init_weights(net, init_type='xavier', init_gain=1.0):
    """
    Initialize network weights.
    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.
    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(ini.initializer(
                    ini.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(ini.initializer(
                    ini.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(
                    ini.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d)):
            cell.gamma.set_data(ini.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(ini.initializer('zeros', cell.beta.shape))
            cell.moving_mean.set_data(ini.initializer(
                'zeros', cell.moving_mean.shape))
            cell.moving_variance.set_data(
                ini.initializer('ones', cell.moving_variance.shape))
