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
#-*- coding:utf-8 -*-

class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super(L1MuLoss, self).__init__()
        self.mu = mu

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return nn.L1Loss()(mu_pred, mu_label)



class JointReconPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000):
        super(JointReconPerceptualLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1MuLoss(self.mu)
        self.loss_vgg = VGGPerceptualLoss(False)

    def forward(self, input1, target):
        input_mu = range_compressor(input1, self.mu)
        target_mu = range_compressor(target, self.mu)
        loss_recon = self.loss_recon(input1, target)
        loss_vgg = self.loss_vgg(input_mu, target_mu)
        loss = loss_recon + self.alpha * loss_vgg
        return loss
