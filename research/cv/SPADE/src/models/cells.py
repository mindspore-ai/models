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
""" define training network """

import mindspore.nn as nn
from mindspore.ops import stop_gradient
import mindspore.ops as ops
import mindspore.ops.composite as C
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size

class GenWithLossCell(nn.Cell):
    """Generator with loss(wrapped)"""

    def __init__(self, opt, netG, netD, GLoss, FLoss, VggLoss):
        super(GenWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD
        self.GLoss = GLoss
        self.FLoss = FLoss
        self.VggLoss = VggLoss
        self.lambda_feat = opt.lambda_feat
        self.lambda_vgg = opt.lambda_vgg

        self.cat_0 = ops.Concat(axis=0)
        self.cat_1 = ops.Concat(axis=1)

    def construct(self, input_semantics, real_image):
        fake_image = self.netG(input_semantics)
        fake_concat = self.cat_1((input_semantics, fake_image))
        real_concat = self.cat_1((input_semantics, real_image))
        fake_and_real = self.cat_0((fake_concat, real_concat))
        discriminator_out = self.netD(fake_and_real)
        pred_fake = []
        pred_real = []
        for p in discriminator_out:
            tmp_fake = []
            tmp_real = []
            for tensor in p:
                tmp = tensor[:tensor.shape[0] // 2]
                tmp_fake.append(tmp)
            pred_fake.append(tmp_fake)
            for tensor in p:
                tmp = tensor[tensor.shape[0] // 2:]
                tmp_real.append(tmp)
            pred_real.append(tmp_real)
        G_losses_GAN = self.GLoss(pred_fake, True, for_discriminator=False)
        GAN_Feat_loss = 0
        num_D = len(pred_fake)
        pred_real = stop_gradient(pred_real)
        for i in range(num_D):
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.FLoss(
                    pred_fake[i][j], pred_real[i][j])
                GAN_Feat_loss += unweighted_loss * self.lambda_feat / num_D
        G_losses_VGG = self.VggLoss(fake_image, real_image) * self.lambda_vgg
        mean_loss = G_losses_GAN + GAN_Feat_loss + G_losses_VGG
        return mean_loss


class GenTrainOneStepCell(nn.Cell):
    """ Generator TrainOneStepCell """

    def __init__(self, G,
                 optimizer: nn.Optimizer,
                 sens=1.0):
        super(GenTrainOneStepCell, self).__init__()
        self.optimizer = optimizer
        self.G = G
        self.G.set_train()
        self.G.set_grad()
        self.G.VggLoss.set_grad(False)
        self.G.VggLoss.set_train(False)

        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, input_semantics, real_image):
        weights = self.weights
        lg = self.G(input_semantics, real_image)
        sens_g = ops.Fill()(ops.DType()(lg), ops.Shape()(lg), self.sens)
        grads_g = self.grad(self.G, weights)(input_semantics, real_image, sens_g)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_g = self.grad_reducer(grads_g)
        lg = ops.depend(lg, self.optimizer(grads_g))
        return lg


class DisWithLossCell(nn.Cell):
    """Generator with loss(wrapped)"""

    def __init__(self, netG, netD, GLoss):
        super(DisWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD
        self.GLoss = GLoss
        self.cat_0 = ops.Concat(axis=0)
        self.cat_1 = ops.Concat(axis=1)

    def construct(self, input_semantics, real_image):
        fake_image = self.netG(input_semantics)
        fake_concat = self.cat_1((input_semantics, fake_image))
        real_concat = self.cat_1((input_semantics, real_image))
        fake_and_real = self.cat_0((fake_concat, real_concat))
        discriminator_out = self.netD(fake_and_real)
        pred_fake = []
        pred_real = []
        for p in discriminator_out:
            tmp_fake = []
            tmp_real = []
            for tensor in p:
                tmp = tensor[:tensor.shape[0] // 2]
                tmp_fake.append(tmp)
            pred_fake.append(tmp_fake)
            for tensor in p:
                tmp = tensor[tensor.shape[0] // 2:]
                tmp_real.append(tmp)
            pred_real.append(tmp_real)
        D_losses_D_Fake = self.GLoss(pred_fake, False,
                                     for_discriminator=True)
        D_losses_D_real = self.GLoss(pred_real, True,
                                     for_discriminator=True)
        mean_loss = D_losses_D_Fake+D_losses_D_real
        return mean_loss


class DisTrainOneStepCell(nn.Cell):
    """ Generator TrainOneStepCell """

    def __init__(self, D,
                 optimizer: nn.Optimizer,
                 sens=1.0):
        super(DisTrainOneStepCell, self).__init__()
        self.optimizer = optimizer
        self.D = D
        self.D.set_train()
        self.D.set_grad()

        self.D.netG.set_train(False)
        self.D.netG.set_grad(False)
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

        self.weights = optimizer.parameters
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, input_semantics, real_image):
        weights = self.weights
        ld = self.D(input_semantics, real_image)
        sens_d = ops.Fill()(ops.DType()(ld), ops.Shape()(ld), self.sens)
        grads_d = self.grad(self.D, weights)(input_semantics, real_image, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_d = self.grad_reducer(grads_d)
        ld = ops.depend(ld, self.optimizer(grads_d))
        return ld
