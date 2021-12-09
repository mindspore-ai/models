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
"""
The Discriminator of APDrawingGAN
"""
import random
import mindspore
from mindspore import nn, ops
from src.networks import controller as networks


class Discriminator(nn.Cell):
    """Discriminator"""
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.support_non_tensor_inputs = True
        self.discriminator_local = opt.discriminator_local
        self.pool_size = opt.pool_size
        self.isTrain = opt.isTrain
        self.fineSize = opt.fineSize
        self.output_nc = opt.output_nc
        self.EYE_H = opt.EYE_H
        self.EYE_W = opt.EYE_W
        self.NOSE_H = opt.NOSE_H
        self.NOSE_W = opt.NOSE_W
        self.MOUTH_H = opt.MOUTH_H
        self.MOUTH_W = opt.MOUTH_W
        use_sigmoid = opt.no_lsgan
        # global
        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain)
        self.D_network_names = ['D']
        # local
        if self.discriminator_local:
            print('D net use local')
            self.netDLEyel = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain)
            self.netDLEyer = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain)
            self.netDLNose = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain)
            self.netDLMouth = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain)
            self.netDLHair = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain)
            self.netDLBG = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                             opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain)
            self.D_network_names = ['D', 'DLEyel', 'DLEyer', 'DLNose', 'DLMouth', 'DLHair', 'DLBG']
            self.local_names = ['DLEyel', 'DLEyer', 'DLNose', 'DLMouth', 'DLHair', 'DLBG']

    def set_index(self, center):
        """set_index"""
        ratio = self.fineSize // 256
        EYE_H = self.EYE_H * ratio
        EYE_W = self.EYE_W * ratio
        NOSE_H = self.NOSE_H * ratio
        NOSE_W = self.NOSE_W * ratio
        MOUTH_H = self.MOUTH_H * ratio
        MOUTH_W = self.MOUTH_W * ratio
        self.eyel1 = int(center[0, 1]) - EYE_H // 2
        self.eyel2 = int(center[0, 1]) + EYE_H // 2
        self.eyel3 = int(center[0, 0] - EYE_W // 2)
        self.eyel4 = int(center[0, 0]) + EYE_W // 2
        self.eyer1 = int(center[1, 1]) - EYE_H // 2
        self.eyer2 = int(center[1, 1] + EYE_H // 2)
        self.eyer3 = int(center[1, 0] - EYE_W // 2)
        self.eyer4 = int(center[1, 0] + EYE_W // 2)
        self.nose1 = int(center[2, 1] - NOSE_H // 2)
        self.nose2 = int(center[2, 1] + NOSE_H // 2)
        self.nose3 = int(center[2, 0] - NOSE_W // 2)
        self.nose4 = int(center[2, 0] + NOSE_W // 2)
        self.mouth1 = int(center[3, 1] - MOUTH_H // 2)
        self.mouth2 = int(center[3, 1] + MOUTH_H // 2)
        self.mouth3 = int(center[3, 0] - MOUTH_W // 2)
        self.mouth4 = int(center[3, 0] + MOUTH_W // 2)

    def _getLocalParts(self, fakeAB, mask, mask2):
        """_getLocalParts"""
        bs, nc, _, _ = fakeAB.shape
        ncr = nc // self.output_nc
        ratio = self.fineSize // 256
        EYE_H = self.EYE_H * ratio
        EYE_W = self.EYE_W * ratio
        NOSE_H = self.NOSE_H * ratio
        NOSE_W = self.NOSE_W * ratio
        MOUTH_H = self.MOUTH_H * ratio
        MOUTH_W = self.MOUTH_W * ratio
        ones = ops.Ones()
        eyel = ones((bs, nc, EYE_H, EYE_W), mindspore.float32)
        eyer = ones((bs, nc, EYE_H, EYE_W), mindspore.float32)
        nose = ones((bs, nc, NOSE_H, NOSE_W), mindspore.float32)
        mouth = ones((bs, nc, MOUTH_H, MOUTH_W), mindspore.float32)

        for i in range(bs):
            eyel[i] = fakeAB[i, :, self.eyel1:self.eyel2, self.eyel3:self.eyel4]
            eyer[i] = fakeAB[i, :, self.eyer1:self.eyer2, self.eyer3:self.eyer4]
            nose[i] = fakeAB[i, :, self.nose1:self.nose2, self.nose3:self.nose4]
            mouth[i] = fakeAB[i, :, self.mouth1:self.mouth2, self.mouth3:self.mouth4]
        tile = ops.Tile()
        hair = (fakeAB / 2 + 0.5) * tile(mask, (1, ncr, 1, 1)) * tile(mask2, (1, ncr, 1, 1)) * 2 - 1
        bg = (fakeAB / 2 + 0.5) * (ones(fakeAB.shape, mindspore.float32) - tile(mask2, (1, ncr, 1, 1))) * 2 - 1
        return eyel, eyer, nose, mouth, hair, bg

    def set_Grad(self, value):
        """set gradient"""
        self.netD.set_grad(value)
        if self.discriminator_local:
            self.netDLEyer.set_grad(value)
            self.netDLEyel.set_grad(value)
            self.netDLMouth.set_grad(value)
            self.netDLNose.set_grad(value)
            self.netDLHair.set_grad(value)
            self.netDLBG.set_grad(value)
        return True

    def pool_query(self, pool_size, Images):
        """pool_query"""
        if pool_size == 0:
            return Images
        return_images = []
        num_imgs = 0
        images = []
        for image in Images:
            unsqueeze = ops.ExpandDims()
            image = unsqueeze(image, 0)
            if num_imgs < self.pool_size:
                num_imgs = num_imgs + 1
                images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, pool_size - 1)  # randint is inclusive
                    tmp = images[random_id].clone()
                    images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        cat = ops.Concat(0)
        return_images = cat((return_images))
        return return_images

    def judge_network_isnan(self):
        """judge_network_isnan"""
        net = self.netD
        for _, prams in net.parameters_and_names():
            _ = prams * 1.00
        if self.discriminator_local:
            net = self.netDLEyel
            for _, prams in net.parameters_and_names():
                _ = prams * 1.00
            net = self.netDLEyer
            for _, prams in net.parameters_and_names():
                _ = prams * 1.00
            net = self.netDLNose
            for _, prams in net.parameters_and_names():
                _ = prams * 1.00
            net = self.netDLMouth
            for _, prams in net.parameters_and_names():
                _ = prams * 1.00
            net = self.netDLHair
            for _, prams in net.parameters_and_names():
                _ = prams * 1.00
            net = self.netDLBG
            for _, prams in net.parameters_and_names():
                _ = prams * 1.00

    def construct(self, A, B, is_fake, mask, mask2):
        """construct"""
        cat = ops.Concat(1)
        if is_fake:
            temp = cat((A, B))
            AB = self.pool_query(self.pool_size, temp)
        else:
            AB = cat((A, B))

        if self.discriminator_local:
            AB_parts = self._getLocalParts(AB, mask, mask2)
            return self.netD(AB), self.netDLEyel(AB_parts[0]), self.netDLEyer(AB_parts[1]), \
                   self.netDLNose(AB_parts[2]), self.netDLMouth(AB_parts[3]), self.netDLHair(AB_parts[4]), \
                   self.netDLBG(AB_parts[5])
        return self.netD(AB)
