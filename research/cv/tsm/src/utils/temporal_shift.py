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
"""temporal_shift"""
import mindspore.nn as nn
import mindspore.ops as ops



from src.model.resnet import ResNet


class TemporalShift(nn.Cell):
    """temporalshift"""
    def __init__(self, net, n_segment=3, n_div=8, layer=1, times=0, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.layer = layer
        self.times = times
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def construct(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        """shift"""
        nt, c, h, w = x.shape
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        op1 = ops.Concat(1)
        op2 = ops.Concat(2)
        zeros = ops.Zeros()
        data_type = x.dtype

        pad = zeros((n_batch, 1, fold, h, w), data_type)
        first_group = op1((x[:, 1:, :fold, :, :], pad))
        second_group = op1((pad, x[:, :n_segment-1, fold:2*fold, :, :]))

        third_group = x[:, :, 2*fold:, :, :]
        out = op2((first_group, second_group, third_group))

        return out.view(nt, c, h, w)


class TemporalPool(nn.Cell):
    """TemporalPool"""
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def construct(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.shape
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = ops.MaxPool3D(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    """make_temporal_shift"""
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    # if isinstance(net, ResNet):
    if place == 'block':
        def make_block_temporal(stage, this_segment):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks'.format(len(blocks)))
            for i, b in enumerate(blocks):
                blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
            return nn.SequentialCell(blocks)

        net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
        net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
        net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
        net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

    elif 'blockres' in place:
        n_round = 1
        if len(list(net.layer3.cells())) >= 23:
            n_round = 2
            print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(stage, this_segment, layer):
            blocks = list(stage.cells())
            print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div, layer=layer, times=i)
            return nn.SequentialCell(blocks)

        net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], layer=1)
        net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], layer=2)
        net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], layer=3)
        net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], layer=4)



def make_temporal_pool(net, n_segment):
    if isinstance(net, ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError
