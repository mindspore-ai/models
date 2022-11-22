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
"""ipt_post_onnx"""
import numpy as np
import onnxruntime as ort
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn
from mindspore.common import Tensor


def create_session(onnx_checkpoint_path, target_device):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")

    session = ort.InferenceSession(onnx_checkpoint_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names

class IPT_post():
    """ipt_post_onnx"""
    def __init__(self, model, args):
        super(IPT_post, self).__init__()
        self.model = model
        self.args = args
        self.scale_idx = 0
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.transpose = ops.Transpose()
        self.cc_0 = ops.Concat(axis=0)
        self.cc_2 = ops.Concat(axis=2)
        self.cc_3 = ops.Concat(axis=3)
        self.shape_onnx = []
        self.eval_onnx = False

    def forward(self, x, idx, shave=12, batchsize=64, eval_onnx=False):
        """ipt"""
        self.eval_onnx = eval_onnx
        self.idx = idx
        h, w = x.shape[-2:]
        padsize = int(self.args.patch_size)
        shave = int(self.args.patch_size / 4)
        scale = self.args.scale[0]
        h_cut = (h - padsize) % (padsize - shave)
        w_cut = (w - padsize) % (padsize - shave)
        unf_1 = _stride_unfold_(padsize, stride=padsize - shave)
        x_unfold = unf_1.compute(x)
        x_unfold = self.transpose(x_unfold, (1, 0, 2))  # transpose(0,2)
        x_hw_cut = x[:, :, (h - padsize):, (w - padsize):]
        if eval_onnx:
            onnx_file = self.args.pth_path + str(x_hw_cut.shape[0]) + '_' + str(x_hw_cut.shape[1]) + '.onnx'
            session, [x_name, idx_name] = create_session(onnx_file, 'GPU')
            y_hw_cut = session.run(None, {x_name: x_hw_cut.asnumpy(), idx_name: self.idx.asnumpy()})
            y_hw_cut = np.squeeze(y_hw_cut, axis=0)
            y_hw_cut = Tensor(y_hw_cut, ms.float16)
        else:
            self.shape_onnx.append(x_hw_cut.shape)
            y_hw_cut = self.model(x_hw_cut, self.idx)
        x_h_cut = x[:, :, (h - padsize):, :]
        x_w_cut = x[:, :, :, (w - padsize):]
        y_h_cut = self.cut_h_new(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_cut = self.cut_w_new(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        x_h_top = x[:, :, :padsize, :]
        x_w_top = x[:, :, :, :padsize]
        y_h_top = self.cut_h_new(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_top = self.cut_w_new(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        x_unfold = self.reshape(
            x_unfold, (x_unfold.shape[0], -1, padsize, padsize))
        x_range = x_unfold.shape[0] // batchsize + \
            (x_unfold.shape[0] % batchsize != 0)
        for i in range(x_range):
            y_unfold_swap = x_unfold[i * batchsize:(i + 1) * batchsize, :, :, :]
            if i == 0:
                if self.eval_onnx:
                    onnx_file = self.args.pth_path + str(y_unfold_swap.shape[0]) + '_' + str(
                        y_unfold_swap.shape[1]) + '.onnx'
                    session, [x_name, idx_name] = create_session(onnx_file, 'GPU')
                    y_unfold = session.run(None, {x_name: y_unfold_swap.asnumpy(), idx_name: self.idx.asnumpy()})
                    y_unfold = np.squeeze(y_unfold, axis=0)
                    y_unfold = Tensor(y_unfold, ms.float16)
                else:
                    self.shape_onnx.append(y_unfold_swap.shape)
                    y_unfold = self.model(y_unfold_swap, self.idx)
            else:
                if self.eval_onnx:
                    onnx_file = self.args.pth_path + str(y_unfold_swap.shape[0]) + '_' + str(
                        y_unfold_swap.shape[1]) + '.onnx'
                    session, [x_name, idx_name] = create_session(onnx_file, 'GPU')
                    y_unfold = \
                        self.cc_0((y_unfold,
                                   Tensor(np.squeeze(session.run(None, {x_name: y_unfold_swap.asnumpy(),
                                                                        idx_name: self.idx.asnumpy()}),
                                                     axis=0), ms.float16)))
                else:
                    self.shape_onnx.append(y_unfold_swap.shape)
                    y_unfold = self.cc_0((y_unfold, self.model(y_unfold_swap, self.idx)))
        if self.eval_onnx:
            y_unf_shape_0 = y_unfold.shape[0]
            fold_1 = \
                _stride_fold_(padsize * scale, output_shape=((h - h_cut) * scale, (w - w_cut) * scale),
                              stride=padsize * scale - shave * scale)
            y = fold_1.compute(self.transpose(self.reshape(
                y_unfold, (y_unf_shape_0, -1, 1)), (2, 0, 1)))
            if y[:, :, padsize * scale:, :].shape[2] == 0:
                y = y_h_top
            else:
                y = self.cc_2((y_h_top, y[:, :, padsize * scale:, :]))
            if y[:, :, :, padsize * scale:].shape[3] == 0:
                y = y_w_top
            else:
                y = self.cc_3((y_w_top, y[:, :, :, padsize * scale:]))
            y_unfold = y_unfold[:, :, int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                                int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)]
            fold_2 = _stride_fold_(padsize * scale - shave * scale,
                                   output_shape=((h - h_cut - shave) *
                                                 scale, (w - w_cut - shave) * scale),
                                   stride=padsize * scale - shave * scale)
            y_inter = fold_2.compute(self.transpose(self.reshape(
                y_unfold, (y_unf_shape_0, -1, 1)), (2, 0, 1)))
            concat1 = self.cc_2((y[:, :, :int(shave / 2 * scale), \
                                 int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)], y_inter))
            concat2 = self.cc_2((concat1, y[:, :, (h - h_cut) * scale - int(shave / 2 * scale):, \
                                          int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)]))
            concat3 = self.cc_3((y[:, :, :, :int(shave / 2 * scale)], concat2))
            y = self.cc_3((concat3, y[:, :, :, (w - w_cut) * scale - int(shave / 2 * scale):]))
            y = self.cc_2((y[:, :, :y.shape[2] - int((padsize - h_cut) / 2 * scale), :],
                           y_h_cut[:, :, int((padsize - h_cut) / 2 * scale + 0.5):, :]))
            y_w_cat = self.cc_2((y_w_cut[:, :, :y_w_cut.shape[2] - int((padsize - h_cut) / 2 * scale), :],
                                 y_hw_cut[:, :, int((padsize - h_cut) / 2 * scale + 0.5):, :]))
            y = self.cc_3((y[:, :, :, :y.shape[3] - int((padsize - w_cut) / 2 * scale)],
                           y_w_cat[:, :, :, int((padsize - w_cut) / 2 * scale + 0.5):]))
            return y
        return self.shape_onnx


    def cut_h_new(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        """ipt"""
        unf_1 = _stride_unfold_(padsize, stride=padsize - shave)
        x_h_cut_unfold = unf_1.compute(x_h_cut)
        x_h_cut_unfold = self.transpose(x_h_cut_unfold, (1, 0, 2))

        x_h_cut_unfold = self.reshape(
            x_h_cut_unfold, (x_h_cut_unfold.shape[0], -1, padsize, padsize))
        x_range = x_h_cut_unfold.shape[0] // batchsize + \
            (x_h_cut_unfold.shape[0] % batchsize != 0)
        for i in range(x_range):
            y_h_cut_unfold_swap = x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, :, :, :]
            if i == 0:
                if self.eval_onnx:
                    onnx_file = self.args.pth_path + str(y_h_cut_unfold_swap.shape[0]) + '_' + str(
                        y_h_cut_unfold_swap.shape[1]) + '.onnx'
                    session, [x_name, idx_name] = create_session(onnx_file, 'GPU')
                    y_h_cut_unfold = session.run(None, {x_name: y_h_cut_unfold_swap.asnumpy(),
                                                        idx_name: self.idx.asnumpy()})
                    y_h_cut_unfold = np.squeeze(y_h_cut_unfold, axis=0)
                    y_h_cut_unfold = Tensor(y_h_cut_unfold, ms.float16)
                else:
                    self.shape_onnx.append(y_h_cut_unfold_swap.shape)
                    y_h_cut_unfold = self.model(y_h_cut_unfold_swap, self.idx)
            else:
                if self.eval_onnx:
                    onnx_file = self.args.pth_path + str(y_h_cut_unfold_swap.shape[0]) + '_' + str(
                        y_h_cut_unfold_swap.shape[1]) + '.onnx'
                    session, [x_name, idx_name] = create_session(onnx_file, 'GPU')
                    y_h_cut_unfold = \
                        self.cc_0((y_h_cut_unfold,
                                   Tensor(np.squeeze(session.run(None, {x_name: y_h_cut_unfold_swap.asnumpy(),
                                                                        idx_name: self.idx.asnumpy()}),
                                                     axis=0), ms.float16)))
                else:
                    self.shape_onnx.append(y_h_cut_unfold_swap.shape)
                    y_h_cut_unfold = self.cc_0((y_h_cut_unfold, self.model(y_h_cut_unfold_swap, self.idx)))
        y_h_cut_unfold_shape_0 = y_h_cut_unfold.shape[0]
        fold_1 = \
            _stride_fold_(padsize * scale, output_shape=(padsize * scale, (w - w_cut) * scale),
                          stride=padsize * scale - shave * scale)
        y_h_cut = fold_1.compute(self.transpose(self.reshape(
            y_h_cut_unfold, (y_h_cut_unfold_shape_0, -1, 1)), (2, 0, 1)))
        y_h_cut_unfold = y_h_cut_unfold[:, :, :, int(
            shave / 2 * scale):padsize * scale - int(shave / 2 * scale)]
        fold_2 = _stride_fold_((padsize * scale, padsize * scale - shave * scale),
                               output_shape=(padsize * scale,
                                             (w - w_cut - shave) * scale),
                               stride=padsize * scale - shave * scale)
        y_h_cut_inter = fold_2.compute(self.transpose(self.reshape(
            y_h_cut_unfold, (y_h_cut_unfold_shape_0, -1, 1)), (2, 0, 1)))
        concat1 = self.cc_3((y_h_cut[:, :, :, :int(shave / 2 * scale)], y_h_cut_inter))
        y_h_cut = self.cc_3((concat1, y_h_cut[:, :, :, (w - w_cut) * scale - int(shave / 2 * scale):]))
        return y_h_cut

    def cut_w_new(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        """ipt"""
        unf_1 = _stride_unfold_(padsize, stride=padsize - shave)
        x_w_cut_unfold = unf_1.compute(x_w_cut)
        x_w_cut_unfold = self.transpose(x_w_cut_unfold, (1, 0, 2))

        x_w_cut_unfold = self.reshape(
            x_w_cut_unfold, (x_w_cut_unfold.shape[0], -1, padsize, padsize))
        x_range = x_w_cut_unfold.shape[0] // batchsize + \
            (x_w_cut_unfold.shape[0] % batchsize != 0)
        for i in range(x_range):
            y_w_cut_unfold_swap = x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, :, :, :]
            if i == 0:
                if self.eval_onnx:
                    onnx_file = self.args.pth_path + str(y_w_cut_unfold_swap.shape[0]) + '_' + str(
                        y_w_cut_unfold_swap.shape[1]) + '.onnx'
                    session, [x_name, idx_name] = create_session(onnx_file, 'GPU')
                    y_w_cut_unfold = session.run(None, {x_name: y_w_cut_unfold_swap.asnumpy(),
                                                        idx_name: self.idx.asnumpy()})
                    y_w_cut_unfold = np.squeeze(y_w_cut_unfold, axis=0)
                    y_w_cut_unfold = Tensor(y_w_cut_unfold, ms.float16)
                else:
                    self.shape_onnx.append(y_w_cut_unfold_swap.shape)
                    y_w_cut_unfold = self.model(y_w_cut_unfold_swap, self.idx)
            else:
                if self.eval_onnx:
                    onnx_file = self.args.pth_path + str(y_w_cut_unfold_swap.shape[0]) + '_' + str(
                        y_w_cut_unfold_swap.shape[1]) + '.onnx'
                    session, [x_name, idx_name] = create_session(onnx_file, 'GPU')
                    y_w_cut_unfold = \
                        self.cc_0((y_w_cut_unfold,
                                   Tensor(np.squeeze(session.run(None, {x_name: y_w_cut_unfold_swap.asnumpy(),
                                                                        idx_name: self.idx.asnumpy()}), axis=0),
                                          ms.float16)))
                else:
                    self.shape_onnx.append(y_w_cut_unfold_swap.shape)
                    y_w_cut_unfold = self.cc_0((y_w_cut_unfold, self.model(y_w_cut_unfold_swap, self.idx)))
        y_w_cut_unfold_shape_0 = y_w_cut_unfold.shape[0]
        fold_1 = _stride_fold_(padsize * scale,
                               output_shape=((h - h_cut) * scale,
                                             padsize * scale),
                               stride=padsize * scale - shave * scale)
        y_w_cut = fold_1.compute(self.transpose(self.reshape(
            y_w_cut_unfold, (y_w_cut_unfold_shape_0, -1, 1)), (2, 0, 1)))
        y_w_cut_unfold = y_w_cut_unfold[:, :, int(
            shave / 2 * scale):padsize * scale - int(shave / 2 * scale), :]
        fold_2 = _stride_fold_((padsize * scale - shave * scale, padsize * scale),
                               output_shape=((h - h_cut - shave)
                                             * scale, padsize * scale),
                               stride=padsize * scale - shave * scale)
        y_w_cut_inter = fold_2.compute(self.transpose(self.reshape(
            y_w_cut_unfold, (y_w_cut_unfold_shape_0, -1, 1)), (2, 0, 1)))
        concat1 = self.cc_2((y_w_cut[:, :, :int(shave / 2 * scale), :], y_w_cut_inter))
        y_w_cut = self.cc_2((concat1, y_w_cut[:, :, (h - h_cut) * scale - int(shave / 2 * scale):, :]))
        return y_w_cut


class _stride_unfold_():
    '''stride'''
    def __init__(self,
                 kernel_size,
                 stride=-1):

        super(_stride_unfold_, self).__init__()
        if stride == -1:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.kernel_size = kernel_size

        self.unfold = _unfold_(kernel_size)

    def compute(self, x):
        """stride"""
        x = x.asnumpy()
        N, C, H, W = x.shape
        leftup_idx_x = []
        leftup_idx_y = []
        nh = (H - self.kernel_size) // self.stride + 1
        nw = (W - self.kernel_size) // self.stride + 1
        for i in range(nh):
            leftup_idx_x.append(i * self.stride)
        for i in range(nw):
            leftup_idx_y.append(i * self.stride)
        NumBlock_x = len(leftup_idx_x)
        NumBlock_y = len(leftup_idx_y)
        unf_x = np.zeros((N, C, NumBlock_x * self.kernel_size, NumBlock_y * self.kernel_size), dtype=np.float32)
        N, C, H, W = unf_x.shape
        for i in range(NumBlock_x):
            for j in range(NumBlock_y):
                unf_i = i * self.kernel_size
                unf_j = j * self.kernel_size
                org_i = leftup_idx_x[i]
                org_j = leftup_idx_y[j]
                fills = x[:, :, org_i:org_i + self.kernel_size,
                          org_j:org_j + self.kernel_size]
                zeros2 = np.zeros(unf_x[:, :, :unf_i, unf_j:unf_j + self.kernel_size].shape)
                concat1 = np.concatenate((zeros2, fills), axis=2)
                zeros3 = np.zeros(unf_x[:, :, unf_i + self.kernel_size:, unf_j:unf_j + self.kernel_size].shape)
                concat2 = np.concatenate((concat1, zeros3), axis=2)
                zeros1 = np.zeros(unf_x[:, :, :, :unf_j].shape)
                concat3 = np.concatenate((zeros1, concat2), axis=3)
                zeros4 = np.zeros(unf_x[:, :, :, unf_j + self.kernel_size:].shape)
                concat4 = np.concatenate((concat3, zeros4), axis=3)
                unf_x += concat4
        unf_x = Tensor(unf_x, ms.float16)
        y = self.unfold(unf_x)
        return y


class _stride_fold_():
    '''stride'''
    def __init__(self,
                 kernel_size,
                 output_shape=(-1, -1),
                 stride=-1):

        super(_stride_fold_, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]

        if stride == -1:
            self.stride = kernel_size[0]
        else:
            self.stride = stride

        self.output_shape = output_shape

        self.NumBlock_x = (self.output_shape[0] - self.kernel_size[0]) // self.stride + 1
        self.NumBlock_y = (self.output_shape[1] - self.kernel_size[1]) // self.stride + 1
        self.large_shape = [self.NumBlock_x * self.kernel_size[0], self.NumBlock_y * self.kernel_size[1]]
        self.fold = _fold_(self.kernel_size, self.large_shape)

    def compute(self, x):
        """ compute"""
        NumBlock_x = self.NumBlock_x
        NumBlock_y = self.NumBlock_y
        large_x = self.fold(x)
        large_x = large_x.asnumpy()
        N, C, _, _ = large_x.shape
        leftup_idx_x = []
        leftup_idx_y = []
        for i in range(NumBlock_x):
            leftup_idx_x.append(i * self.kernel_size[0])
        for i in range(NumBlock_y):
            leftup_idx_y.append(i * self.kernel_size[1])
        fold_x = np.zeros((N, C, (NumBlock_x - 1) * self.stride + self.kernel_size[0], \
                                          (NumBlock_y - 1) * self.stride + self.kernel_size[1]), dtype=np.float32)
        for i in range(NumBlock_x):
            for j in range(NumBlock_y):
                fold_i = i * self.stride
                fold_j = j * self.stride
                org_i = leftup_idx_x[i]
                org_j = leftup_idx_y[j]
                fills = large_x[:, :, org_i:org_i + self.kernel_size[0], org_j:org_j + self.kernel_size[1]]
                t2 = fold_x[:, :, :fold_i, fold_j:fold_j + self.kernel_size[1]]
                zeros2 = np.zeros(t2.shape)
                concat1 = np.concatenate((zeros2, fills), axis=2)
                t3 = fold_x[:, :, fold_i + self.kernel_size[0]:, fold_j:fold_j + self.kernel_size[1]]
                zeros3 = np.zeros(t3.shape)
                concat2 = np.concatenate((concat1, zeros3), axis=2)
                t1 = fold_x[:, :, :, :fold_j]
                zeros1 = np.zeros(t1.shape)
                concat3 = np.concatenate((zeros1, concat2), axis=3)
                t4 = fold_x[:, :, :, fold_j + self.kernel_size[1]:]
                zeros4 = np.zeros(t4.shape)
                concat4 = np.concatenate((concat3, zeros4), axis=3)
                fold_x += concat4
        y = Tensor(fold_x, ms.float16)
        return y

class _unfold_(nn.Cell):
    """ipt"""
    def __init__(
            self, kernel_size, stride=-1):

        super(_unfold_, self).__init__()
        if stride == -1:
            self.stride = kernel_size
        self.kernel_size = kernel_size

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        """ipt"""
        N, C, H, W = x.shape
        numH = H // self.kernel_size
        numW = W // self.kernel_size
        if numH * self.kernel_size != H or numW * self.kernel_size != W:
            x = x[:, :, :numH * self.kernel_size, :, numW * self.kernel_size]
        output_img = self.reshape(x, (N, C, numH, self.kernel_size, W))

        output_img = self.transpose(output_img, (0, 1, 2, 4, 3))
        output_img = self.reshape(output_img, (N*C, numH, numW, self.kernel_size, self.kernel_size))
        output_img = self.transpose(output_img, (0, 1, 2, 4, 3))
        output_img = self.reshape(output_img, (N, C, numH * numW, self.kernel_size*self.kernel_size))
        output_img = self.transpose(output_img, (0, 2, 1, 3))
        output_img = self.reshape(output_img, (N, numH * numW, -1))
        return output_img


class _fold_(nn.Cell):
    """ipt"""

    def __init__(
            self, kernel_size, output_shape=(-1, -1), stride=-1):

        super(_fold_, self).__init__()

        # if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]

        if stride == -1:
            self.stride = self.kernel_size[0]
        self.output_shape = output_shape

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.sqrt = ops.Sqrt()
        self.cast = ops.Cast()

    def construct(self, x):
        """ipt"""
        N, C, L = x.shape
        org_C = L // (self.kernel_size[0] * self.kernel_size[1])
        org_H = self.output_shape[0]
        org_W = self.output_shape[1]
        numH = org_H // self.kernel_size[0]
        numW = org_W // self.kernel_size[1]
        output_img = self.reshape(x, (N, C, org_C, self.kernel_size[0], self.kernel_size[1]))
        output_img = self.transpose(output_img, (0, 2, 3, 1, 4))
        output_img = self.reshape(output_img, (N*org_C, self.kernel_size[0], numH, numW, self.kernel_size[1]))
        output_img = self.transpose(output_img, (0, 2, 1, 3, 4))

        output_img = self.reshape(output_img, (N, org_C, org_H, org_W))
        return output_img
