# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import numpy
import mindspore
import mindspore.numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import One
from sklearn.cluster import KMeans
import res20_adder

class Round(nn.Cell):
    def __init__(self):
        super(Round, self).__init__()
        self.round = ops.Round()

    def construct(self, inp):
        return self.round(inp)

def original_order(ordered, indices):
    z = np.ones_like(ordered)
    for i in range(ordered.shape[1]):
        z[:, int(indices[i]), :, :] = ordered[:, i, :, :]
    return z

def unfold(img, kernel_size, stride=1, pad=0, dilation=1):
    """
    unfold function
    """
    batch_num, channel, height, width = img.shape
    out_h = (height + pad + pad - kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1
    out_w = (width + pad + pad - kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1

    img = np.pad(img, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((batch_num, channel, kernel_size, kernel_size, out_h, out_w)).astype(img.dtype)

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = np.reshape(col, (batch_num, channel * kernel_size * kernel_size, out_h * out_w))

    return col

class Adder(nn.Cell):
    def __init__(self):
        super(Adder, self).__init__()
        self.abs = ops.Abs()
        self.sum = ops.ReduceSum(keep_dims=False)
        self.expand_dims = ops.ExpandDims()
        self.lp_norm = ops.LpNorm(axis=[0, 1], p=2, keep_dims=False)
        self.sqrt = ops.Sqrt()

    def construct(self, W_col, X_col):
        output = -self.sum(self.abs((self.expand_dims(W_col, 2)-self.expand_dims(X_col, 0))), 1)
        return output


def adder2d_function(X, W, stride=1, padding=0):
    n_filters, _, h_filter, w_filter = W.shape
    n_x, _, h_x, w_x = X.shape

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    # here rates padding
    X_col = unfold((X.view(
        1, -1, h_x, w_x)), kernel_size=h_filter, stride=stride, pad=padding, dilation=1).view(n_x, -1, h_out * w_out)
    adder = Adder()

    #here ravel
    X_col = ops.transpose(X_col, (1, 2, 0)).view(X_col.shape[1], -1)
    W_col = W.view(n_filters, -1)

    out = adder(W_col, X_col)
    out = out.view(n_filters, h_out, w_out, n_x)
    out = ops.transpose(out, (3, 0, 1, 2))
    return out


class QAdder2dKmeansGroupShare(nn.Cell):
    def __init__(
            self, in_channels, output_channel, nbit_a, nbit_w, quant_group, kernel_size, stride=1,
                padding=0, bias=False):
        super(QAdder2dKmeansGroupShare, self).__init__()
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.counting = True
        self.nbit_a = nbit_a
        self.nbit_w = nbit_w
        self.quant_group = quant_group
        self.cluster_id = [] #2d
        self.cluster_id_1d = []
        self.bias_list = []
        self.bias_index = []
        self.a_select_range = 0.9997
        self.a_select_flag = 1

        self.act_range = mindspore.Tensor((-1000,), dtype=mindspore.float32)
        scale_para = mindspore.Tensor(shape=(quant_group,), dtype=mindspore.float32, init=One())
        self.scale_para = mindspore.Parameter(scale_para)
        self.sort = ops.Sort()
        self.round = Round()
        self.expand_dims = ops.ExpandDims()

        weight_shape = (output_channel, in_channels, kernel_size, kernel_size)
        weight = Tensor(res20_adder.kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
        self.adder = Parameter(weight)
        self.adder_new = Parameter(weight)

    def cluster_by_kmeans(self):
        param_np = self.adder.value().asnumpy() #modify here
        w = numpy.reshape(param_np, (param_np.shape[0], -1))
        # print('cluster {} filters into {} clusters'.format(x.shape[0], self.quant_group))
        w_sort = numpy.sort(numpy.absolute(w))
        w_select_len = round(len(w[0])*1)-1
        w_max = w_sort[:, w_select_len]
        w_max = numpy.expand_dims(w_max, axis=1)
        km = KMeans(n_clusters=self.quant_group, init='k-means++', max_iter=300, tol=1e-4)
        km.fit(w_max)

        cluster_id_tmp = []
        for _ in range(self.quant_group):
            cluster_id_tmp.append([])
        for i, c in enumerate(km.labels_): #c in quant_group, i in cout
            cluster_id_tmp[c].append(i)
        for r in cluster_id_tmp:
            assert r
        self.cluster_id = cluster_id_tmp

        for i in range(self.quant_group):
            for idx in self.cluster_id[i]:
                self.cluster_id_1d.append(idx)
        self.cluster_id_1d = numpy.array(self.cluster_id_1d)

    def set_para(self):
        if self.act_range < 1e-3:
            self.act_range = Tensor((1e-3,)).copy()
        for idx in range(self.quant_group):
            w = ops.gather(self.adder, Tensor(self.cluster_id[idx], mindspore.int32), 0)
            start_index = 0
            end_index = len(self.cluster_id[0])
            for inner_idx in range(idx):
                start_index += len(self.cluster_id[inner_idx])
                end_index += len(self.cluster_id[inner_idx+1])
            if w.abs().max() > self.act_range[0]:
                self.scale_para[idx] = self.act_range[0].copy()
                extra_bias = -ops.clip_by_value(w.abs() - self.scale_para.data[idx], 0, 1e10).sum(axis=(1, 2, 3))
                self.bias_list.append(Parameter(Tensor(extra_bias), requires_grad=True))
                self.bias_index.append(idx)
                clip_weight = ops.clip_by_value(w, -self.scale_para[idx], self.scale_para[idx]) #return tensor scalar,ranther than python scala
                self.adder_new[start_index:end_index] = clip_weight.copy()
            else:
                self.scale_para[idx] = w.copy().abs().max()
                self.adder_new[start_index:end_index] = w.copy()

    def construct(self, x):
        if self.counting:
            if self.a_select_flag == 1:
                input_temp = x.view(-1)
                input_sorted, _ = self.sort(input_temp.abs())
                a_select_len = int((round(len(input_sorted)*self.a_select_range)-1)*self.a_select_range)
                max_a = input_sorted[a_select_len]
            else:
                max_a = x.abs().max()
            max_new = ops.maximum(self.act_range, max_a)
            self.act_range = max_new[:]
            output = adder2d_function(x, self.adder, self.stride, self.padding)
            return output


        output_list = []
        bias_check = 0
        for idx in range(self.quant_group):
            start_index = 0
            end_index = len(self.cluster_id[0])
            for inner_idx in range(idx):
                start_index += len(self.cluster_id[inner_idx])
                end_index += len(self.cluster_id[inner_idx+1])

            wq = self.adder_new[start_index:end_index] * ((
                2.0**self.nbit_w)-1) / self.scale_para[idx].view(-1, 1, 1, 1) / 2
            wq = self.round(wq)
            wq = ops.clip_by_value(wq, -2.0**(self.nbit_w-1), (2.0**(self.nbit_w-1)) - 1)
            w = wq * 2 * self.scale_para[idx].view(-1, 1, 1, 1) / ((2.0**self.nbit_w)-1)

            aq = x * ((2.0**(self.nbit_a))-1) / self.scale_para[idx]
            aq = self.round(aq)
            aq = ops.clip_by_value(aq, 0, (2.0**self.nbit_a) - 1)
            a = aq * self.scale_para[idx] / ((2.0**(self.nbit_a))-1)
            output = adder2d_function(a, w, self.stride, self.padding)

            if idx in self.bias_index:
                output += self.expand_dims(self.expand_dims(self.expand_dims(self.bias_list[bias_check], 0), 2), 3)
                bias_check += 1
            output_list.append(output)

            output = ops.concat(output_list, axis=1)
            output_final = original_order(output, self.cluster_id_1d)
            return output_final
