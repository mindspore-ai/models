# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore.common.dtype as mstype

from layers.GradientHighwayUnit import GHU as ghu
from layers.CausalLSTMCell import CausalLSTMCell as cslstm

class PreRNN(nn.Cell):
    def __init__(self,
                 input_shape,
                 num_layers,
                 num_hidden,
                 filter_size,
                 stride=1,
                 seq_length=20,
                 input_length=10,
                 tln=True):
        super(PreRNN, self).__init__()
        self.lstm = nn.CellList()

        self.output_channels = input_shape[-3]
        self.batch_size = input_shape[0]

        self.seq_length = seq_length
        self.input_length = input_length
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        for i in range(num_layers):
            if i == 0:
                num_hidden_in = num_hidden[num_layers - 1]
            else:
                num_hidden_in = num_hidden[i - 1]
            new_cell = cslstm('lstm_' + str(i + 1),
                              filter_size,
                              num_hidden_in,
                              num_hidden[i],
                              num_hidden[0],
                              input_shape,
                              tln=tln)
            self.lstm.append(new_cell)

        self.gradient_highway = ghu('highway', input_shape, filter_size, num_hidden[0], tln=tln)

        self.conv2d = nn.Conv2d(num_hidden[3], self.output_channels, 1, 1, padding=0, pad_mode='same')

        self.pack = P.Pack()
        self.transpose = P.Transpose()
        self.print = P.Print()

    def construct(self, images, mask_true):

        gen_images = []
        pre_hidden = []
        hidden = []

        pre_cell = []
        cell = []

        mem = None
        z_t = None
        x_gen = None

        for t in range(self.seq_length - 1):

            if t < self.input_length:
                inputs = images[:, t]
            else:
                inputs = mask_true[:, t - self.input_length] * images[:, t] + \
                (1 - mask_true[:, t - self.input_length]) * x_gen

            if t == 0:
                hidden_input_0 = None
                cell_input_0 = None
            else:
                hidden_input_0 = pre_hidden[0]
                cell_input_0 = pre_cell[0]

            hidden_out_0, cell_out_0, mem = self.lstm[0](inputs, hidden_input_0, cell_input_0, mem)
            hidden.append(hidden_out_0)
            cell.append(cell_out_0)

            z_t = self.gradient_highway(hidden[0], z_t)
            if t == 0:
                hidden_input_1 = None
                cell_input_1 = None
            else:
                hidden_input_1 = pre_hidden[1]
                cell_input_1 = pre_cell[1]

            hidden_out_1, cell_out_1, mem = self.lstm[1](z_t, hidden_input_1, cell_input_1, mem)
            hidden.append(hidden_out_1)
            cell.append(cell_out_1)

            for i in range(2, self.num_layers):
                if t == 0:
                    hidden_input_i = None
                    cell_input_i = None
                else:
                    hidden_input_i = pre_hidden[i]
                    cell_input_i = pre_cell[i]
                hidden_out_i, cell_out_i, mem = self.lstm[i](hidden[i - 1], hidden_input_i, cell_input_i, mem)
                hidden.append(hidden_out_i)
                cell.append(cell_out_i)

            x_gen = self.conv2d(hidden[self.num_layers - 1])

            gen_images.append(x_gen)

            pre_hidden = hidden
            hidden = []

            pre_cell = cell
            cell = []

        gen_images = self.pack(gen_images)
        gen_images = self.transpose(gen_images, (1, 0, 2, 3, 4))

        return gen_images

class NetWithLossCell(nn.Cell):
    def __init__(self, network, batch_size=8, seq_length=20, input_length=10, img_channel=1, img_width=64,
                 reverse_input=True, is_training=True):
        super(NetWithLossCell, self).__init__()

        self.network = network
        self.loss = P.L2Loss()
        self.batch_size = batch_size
        self.is_training = is_training
        self.assignadd = P.AssignAdd()
        downscale = 4
        img_width = img_width * downscale
        self.cur_step = ms.Parameter(ms.Tensor([1], ms.int32), name='global_step1', requires_grad=False)
        self.eta = ms.Parameter(ms.Tensor([1], ms.float32), name='eta', requires_grad=False)
        self.print = P.Print()
        self.minval = ms.Tensor(0, mstype.float32)
        self.maxval = ms.Tensor(1, mstype.float32)
        self.shape = (batch_size, seq_length - input_length - 1)
        self.greater = P.Greater()
        self.div = P.Div()

        self.relu = P.ReLU()
        self.ceil_op = P.Ceil()
        self.broad_shape = (self.batch_size, seq_length - input_length - 1, 1, 1, 1)

        patch_size = 4
        if seq_length - input_length - 1 == 0:
            self.mask_true_shape = (self.batch_size, 1, img_channel, int(img_width/patch_size), \
                int(img_width/patch_size))
        else:
            self.mask_true_shape = (self.batch_size, seq_length - input_length - 1, img_channel, \
                int(img_width/patch_size), int(img_width/patch_size))
        self.broadcast_to = P.BroadcastTo(
            (self.batch_size, seq_length - input_length - 1, img_channel, int(img_width/patch_size), \
                int(img_width/patch_size)))

        self.reshape = P.Reshape()
        self.mask_true = ms.Parameter(ms.Tensor(np.zeros(self.mask_true_shape), ms.float32), name='mask_true',
                                      requires_grad=False)
        self.mod = P.Mod()
        self.reverse = P.ReverseV2(axis=[1])
        self.reverse_input = reverse_input

    def construct(self, images):
        self.eta = self.eta - 0.00002

        if self.is_training:

            random_flip = C.uniform(self.shape, self.minval, self.maxval)
            self.eta = self.relu(self.eta)
            true_token = self.ceil_op(self.eta - random_flip)
            mask_true = self.reshape(true_token, self.broad_shape)
            self.mask_true = self.broadcast_to(mask_true)

        gen_images = self.network(images, self.mask_true)

        if self.is_training:
            gen_images = self.loss(gen_images - images[:, 1:])
            gen_images = gen_images / self.batch_size
            self.cur_step = F.depend(self.cur_step, gen_images)
            self.cur_step = self.assignadd(self.cur_step, 1)

        if self.reverse_input:
            images_rev = self.reverse(images)
            gen_images_rev = self.network(images_rev, self.mask_true)
            second_loss = self.loss(gen_images_rev - images_rev[:, 1:])
            second_loss = second_loss / self.batch_size
            gen_images = (gen_images + second_loss) / 2

        return gen_images

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class ppTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(ppTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.hyper_map = C.HyperMap()

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
