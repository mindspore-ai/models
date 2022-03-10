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
"""
loss function of dien.
"""
from mindspore import nn
from mindspore import ops as P
from mindspore import dtype as mstype


class Ctr_Loss(nn.Cell):
    '''
    negative log-likelihood loss function
    '''

    def __init__(self):
        super(Ctr_Loss, self).__init__()
        self.reducemean = P.ReduceMean()
        self.log = P.Log()

    def construct(self, y_hat, target_ph, aux_loss):
        loss = -self.reducemean(self.log(y_hat) * target_ph) + aux_loss
        return loss


class Auxiliary_Loss(nn.Cell):
    '''
    auxiliary loss function
    '''

    def __init__(self):
        super(Auxiliary_Loss, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=72, eps=1e-3, momentum=0.99)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.cast = P.Cast()
        self.concat_neg1 = P.Concat(axis=-1)
        self.concat0 = P.Concat(axis=0)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.log = P.Log()
        self.reducemean = P.ReduceMean()
        self.denselayer1 = nn.Dense(in_channels=72, out_channels=100, weight_init='xavier_uniform')
        self.denselayer2 = nn.Dense(in_channels=100, out_channels=50, weight_init='xavier_uniform')
        self.denselayer3 = nn.Dense(in_channels=50, out_channels=2, weight_init='xavier_uniform')

    def auxiliary_net(self, in_):
        y_hat = self.bn1(in_)
        y_hat = self.denselayer1(y_hat)
        y_hat = self.sigmoid(y_hat)
        y_hat = self.denselayer2(y_hat)
        y_hat = self.sigmoid(y_hat)
        y_hat = self.denselayer3(y_hat)
        y_hat = self.softmax(y_hat) + 0.00000001
        return y_hat

    def construct(self, h_states, click_seq, noclick_seq, mask):
        mask = self.cast(mask, mstype.float32)
        # concat the end dimension
        click_input_ = self.concat_neg1([h_states, click_seq])
        noclick_input_ = self.concat_neg1([h_states, noclick_seq])
        click_prop_list = []
        noclick_prop_list = []
        for i in range(128):
            click_prop_item = self.auxiliary_net(click_input_[i])
            noclick_prop_item = self.auxiliary_net(noclick_input_[i])
            click_prop_list.append(self.expand_dims(click_prop_item, 0))
            noclick_prop_list.append(self.expand_dims(noclick_prop_item, 0))
        click_prop_ = click_prop_list[0]
        noclick_prop_ = noclick_prop_list[0]
        for i in range(127):
            click_prop_ = self.concat0((click_prop_, click_prop_list[i + 1]))
            noclick_prop_ = self.concat0((noclick_prop_, noclick_prop_list[i + 1]))
        # Get the last y_hat of positive sample
        click_prop_ = click_prop_[:, :, 0]
        # Get the last y_hat of negative sample
        noclick_prop_ = noclick_prop_[:, :, 0]
        # calculate log loss, and mask the real historical behavior
        click_loss_ = -self.reshape(self.log(click_prop_), (-1, self.shape(click_seq)[1])) * mask
        noclick_loss_ = -self.reshape(self.log(1.0 - noclick_prop_), (-1, self.shape(noclick_seq)[1])) * mask
        loss_ = self.reducemean(click_loss_ + noclick_loss_)
        return loss_
