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
from mindspore import nn
import mindspore.ops as ops
from src.public import default_conv, BasicBlock
from src.utils import reshape, reduce_sum


def same_padding(images, ksizes, strides, rates):
    batch_size, channel, rows, cols = images.shape
    batch_size, channel = channel, batch_size
    out_rows = (rows + strides - 1) // strides
    out_cols = (cols + strides - 1) // strides
    effective_k_row = (ksizes - 1) * rates + 1
    effective_k_col = (ksizes - 1) * rates + 1
    temp1 = (out_rows - 1) * strides + effective_k_row - rows
    temp2 = (out_cols - 1) * strides + effective_k_col - cols
    if temp1 < 0:
        temp1 = 0
    if temp2 < 0:
        temp2 = 0
    padding_rows = temp1
    padding_cols = temp2

    padding_top = padding_rows / 2
    padding_left = padding_cols / 2
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left

    paddings = ((0, 0), (0, 0), (padding_top, padding_bottom), (padding_left, padding_right))
    pad = ops.Pad(paddings=paddings)
    images = pad(images)

    return images


def extract_image_patches(unfold, images, ksizes, strides, rates, padding='same'):
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    patches = unfold(images)
    return patches


class NonLocalAttention(nn.Cell):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=default_conv):
        super(NonLocalAttention, self).__init__()
        self.conv_match1 = BasicBlock(conv, channel, channel // reduction, 1, bn=False,
                                      act=nn.PReLU(channel // reduction))
        self.conv_match2 = BasicBlock(conv, channel, channel // reduction, 1, bn=False,
                                      act=nn.PReLU(channel // reduction))
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU(channel))

    def construct(self, inputs):
        x_embed_1 = self.conv_match1(inputs)
        x_embed_2 = self.conv_match2(inputs)
        x_assembly = self.conv_assembly(inputs)

        N, C, H, W = x_embed_1.shape
        x_embed_1 = reshape(x_embed_1.transpose(0, 2, 3, 1), (N, H * W, C))
        x_embed_2 = reshape(x_embed_2, (N, C, H * W))

        score = ops.matmul(x_embed_1, x_embed_2)
        softmax = ops.Softmax(2)
        score = softmax(score)
        x_assembly = reshape(x_assembly, (N, -1, H * W)).transpose(0, 2, 1)
        x_final = ops.matmul(score, x_assembly)
        return reshape(x_final.transpose(0, 2, 1), (N, -1, H, W))


class CrossScaleAttention(nn.Cell):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=default_conv):
        super(CrossScaleAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale
        self.scale = scale
        self.average = average
        self.kernel = self.scale * self.ksize
        self.strides = self.stride * self.scale
        self.escape_NaN = mindspore.Tensor([1e-4], mindspore.float32)  # maybe not
        self.conv_match_1 = BasicBlock(conv, channel, channel // reduction, 1, bn=False,
                                       act=nn.PReLU(channel // reduction))
        self.conv_match_2 = BasicBlock(conv, channel, channel // reduction, 1, bn=False,
                                       act=nn.PReLU(channel // reduction))
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU(channel))
        self.unfold_1 = nn.Unfold(ksizes=[1, self.kernel, self.kernel, 1], strides=[1, self.strides, self.strides, 1],
                                  rates=[1, 1, 1, 1], padding="valid")
        self.unfold_2 = nn.Unfold(ksizes=[1, self.ksize, self.ksize, 1], strides=[1, self.stride, self.stride, 1],
                                  rates=[1, 1, 1, 1], padding="valid")

    def construct(self, inputs):
        embed_w = self.conv_assembly(inputs)
        match_input = self.conv_match_1(inputs)

        shape_input = embed_w.shape
        split = ops.Split(0, match_input.shape[0])
        input_groups = split(match_input)
        kernel = self.scale * self.ksize

        raw_w = extract_image_patches(self.unfold_1, embed_w, kernel, self.stride * self.scale,
                                      rates=1, padding='same')
        raw_w = reshape(raw_w, (shape_input[0], kernel, kernel, shape_input[1], -1))
        transpose = ops.Transpose()
        raw_w = transpose(raw_w, (0, 4, 3, 1, 2))
        split = ops.Split(0, raw_w.shape[0])
        raw_w_groups = split(raw_w)

        reseze_bilinear = ops.ResizeBilinear((inputs.shape[2] // self.scale, inputs.shape[3] // self.scale))
        ref = reseze_bilinear(inputs)
        ref = self.conv_match_2(ref)
        w = extract_image_patches(self.unfold_2, ref, ksizes=self.ksize, strides=self.stride,
                                  rates=1, padding='same')
        shape_ref = ref.shape

        w = reshape(w, (shape_ref[0], self.ksize, self.ksize, shape_ref[1], -1))
        w = w.transpose(0, 4, 3, 1, 2)
        split = ops.Split(0, w.shape[0])
        w_groups = split(w)

        y = []
        scale = self.softmax_scale
        Max = ops.Maximum()
        sqrt = ops.Sqrt()
        Pow = ops.Pow()
        for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
            wi = reshape(wi, (wi.shape[1], wi.shape[2], wi.shape[3], wi.shape[4]))
            max_wi = Max(sqrt(reduce_sum(Pow(wi, 2), axis=[3, 2, 1], keepdim=True)), self.escape_NaN)
            wi_normed = wi / max_wi
            xi = same_padding(xi, self.ksize, 1, 1)
            conv2d = ops.Conv2D(wi_normed.shape[0], wi_normed.shape[3], 1, "valid", 0, 1, 1, 1, "NCHW")
            yi = conv2d(xi, wi_normed)
            yi = reshape(yi, (1, shape_ref[2] * shape_ref[3], shape_input[2], shape_input[3]))  # [1,12*12,24,24]
            softmax = ops.Softmax(1)
            yi = softmax(yi * scale)
            if not self.average:
                reducemax = ops.ReduceMax(True)
                yi = (yi == reducemax(yi, 1))
                type_dst = mindspore.float32
                cast = ops.Cast()
                yi = cast(yi, type_dst)
            wi_center = reshape(raw_wi, (raw_wi.shape[1], raw_wi.shape[2], raw_wi.shape[3], raw_wi.shape[4]))

            conv2d_backprop_input = ops.Conv2DBackpropInput(wi_center.shape[0], wi_center.shape[3], "pad", self.scale,
                                                            None, 1, self.stride * self.scale, 1, 1, "NCHW")
            yi = conv2d_backprop_input(yi, wi_center,
                                       (1, wi_center.shape[1], yi.shape[2] * self.scale, yi.shape[3] * self.scale))
            yi = yi / 6.
            y.append(yi)
        cat = ops.Concat()
        y = cat(y)
        return y
