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
The Generator of APDrawingGAN
"""

from mindspore import nn, ops
import mindspore
from src.networks import controller as networks

class PadOnes(nn.Cell):
    """PadOnes"""
    def __init__(self, margin):
        super(PadOnes, self).__init__()
        self.margin = margin
        self.ones = ops.Ones()
        self.concat_h = ops.Concat(2)
        self.concat_w = ops.Concat(3)

    def construct(self, item):
        bs, nc, h, w = item.shape
        m_top = self.ones((bs, nc, self.margin[0][0], w), mindspore.float32)
        m_down = self.ones((bs, nc, self.margin[0][1], w), mindspore.float32)
        h = h + self.margin[0][0] + self.margin[0][1]
        m_left = self.ones((bs, nc, h, self.margin[1][0]), mindspore.float32)
        m_right = self.ones((bs, nc, h, self.margin[1][1]), mindspore.float32)
        item = self.concat_h((m_top, item, m_down))
        item = self.concat_w((m_left, item, m_right))
        return item

class Generator(nn.Cell):
    """
    Define generator model of APDrawingGAN
    """

    def __init__(self, opt):
        super(Generator, self).__init__()
        # init parameters
        self.support_non_tensor_inputs = True

        self.fineSize = opt.fineSize
        self.which_direction = opt.which_direction
        self.use_local = opt.use_local
        self.isTrain = opt.isTrain
        self.isExport = opt.isExport
        self.comb_op = opt.comb_op
        self.EYE_H = opt.EYE_H
        self.EYE_W = opt.EYE_W
        self.NOSE_H = opt.NOSE_H
        self.NOSE_W = opt.NOSE_W
        self.MOUTH_H = opt.MOUTH_H
        self.MOUTH_W = opt.MOUTH_W

        self.support_non_tensor_inputs = True
        # define Generator
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain,
                                      opt.nnG)
        self.G_network_names = ['G']
        if self.use_local:
            print('G net use local')
            self.netGLEyel = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet', opt.norm,
                                               not opt.no_dropout, opt.init_type, opt.init_gain, 3)
            self.netGLEyer = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet', opt.norm,
                                               not opt.no_dropout, opt.init_type, opt.init_gain, 3)
            self.netGLNose = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet', opt.norm,
                                               not opt.no_dropout, opt.init_type, opt.init_gain, 3)
            self.netGLMouth = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet', opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, 3)
            self.netGLHair = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet2', opt.norm,
                                               not opt.no_dropout, opt.init_type, opt.init_gain, 4)
            self.netGLBG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet2', opt.norm,
                                             not opt.no_dropout, opt.init_type, opt.init_gain, 4)
            self.netGCombine = networks.define_G(2 * opt.output_nc, opt.output_nc, opt.ngf, 'combiner', opt.norm,
                                                 not opt.no_dropout, opt.init_type, opt.init_gain, 2)
            self.G_network_names = ['G', 'GLEyel', 'GLEyer', 'GLNose', 'GLMouth', 'GLHair', 'GLBG', 'GCombine']

    def _addone_with_mask(self, A, mask):
        ones = ops.Ones()
        return ((A / 2 + 0.5) * mask + (ones(mask.shape, mindspore.float32) - mask)) * 2 - 1

    def _masked(self, A, mask):
        return (A / 2 + 0.5) * mask * 2 - 1

    def _partCombiner2_bg(self, eyel, eyer, nose, mouth, hair, bg, maskh, maskb, comb_op=1):
        """
        combine all parts
        """
        if comb_op == 0:
            # use max pooling, pad black for eyes etc
            hair = self._masked(hair, maskh)
            bg = self._masked(bg, maskb)
        else:
            # use min pooling, pad white for eyes etc
            hair = self._addone_with_mask(hair, maskh)
            bg = self._addone_with_mask(bg, maskb)
        eyel_p = self.pad_el(eyel)
        eyer_p = self.pad_er(eyer)
        nose_p = self.pad_no(nose)
        mouth_p = self.pad_mo(mouth)
        if comb_op == 0:
            maximum = ops.Maximum()
            eyes = maximum(eyel_p, eyer_p)
            eye_nose = maximum(eyes, nose_p)
            eye_nose_mouth = maximum(eye_nose, mouth_p)
            eye_nose_mouth_hair = maximum(hair, eye_nose_mouth)
            result = maximum(bg, eye_nose_mouth_hair)
        else:
            minimum = ops.Minimum()
            eyes = minimum(eyel_p, eyer_p)
            eye_nose = minimum(eyes, nose_p)
            eye_nose_mouth = minimum(eye_nose, mouth_p)
            eye_nose_mouth_hair = minimum(hair, eye_nose_mouth)
            result = minimum(bg, eye_nose_mouth_hair)
        return result

    def _inverse_mask(self, mask):
        ones = ops.Ones()
        return ones(mask.shape, mindspore.float32) - mask

    def _generate_output(self, real_A, real_A_bg, real_A_eyel,
                         real_A_eyer, real_A_nose, real_A_mouth,
                         real_A_hair, mask, mask2):
        """
        generate output
        """
        # global
        fake_B0 = self.netG(real_A)
        # local
        if self.use_local:
            fake_B_eyel = self.netGLEyel(real_A_eyel)
            fake_B_eyer = self.netGLEyer(real_A_eyer)
            fake_B_nose = self.netGLNose(real_A_nose)
            fake_B_mouth = self.netGLMouth(real_A_mouth)
            fake_B_hair = self.netGLHair(real_A_hair)
            fake_B_bg = self.netGLBG(real_A_bg)

            fake_B1 = self._partCombiner2_bg(fake_B_eyel, fake_B_eyer, fake_B_nose, fake_B_mouth, fake_B_hair,
                                             fake_B_bg, mask * mask2, self._inverse_mask(mask2),
                                             self.comb_op)
            op = ops.Concat(1)
            output = op((fake_B0, fake_B1))

            fake_B = self.netGCombine(output)
            if self.isExport:
                return fake_B
            return fake_B, fake_B_eyel, fake_B_eyer, fake_B_nose, fake_B_mouth, \
                    self._masked(fake_B_hair, mask * mask2), self._masked(fake_B_bg, self._inverse_mask(mask2))
        return fake_B0

    def set_Grad(self, value):
        self.netG.set_grad(value)
        if self.use_local:
            self.netGLEyer.set_grad(value)
            self.netGLEyel.set_grad(value)
            self.netGLMouth.set_grad(value)
            self.netGLNose.set_grad(value)
            self.netGLHair.set_grad(value)
            self.netGLBG.set_grad(value)
        return True

    def set_pad(self, center):
        """
        set padding function
        """
        IMAGE_SIZE = self.fineSize
        ratio = IMAGE_SIZE / 256
        EYE_W = self.EYE_W * ratio
        EYE_H = self.EYE_H * ratio
        NOSE_W = self.NOSE_W * ratio
        NOSE_H = self.NOSE_H * ratio
        MOUTH_W = self.MOUTH_W * ratio
        MOUTH_H = self.MOUTH_H * ratio
        self.pad_el = PadOnes((
            (int(center[0, 1] - EYE_H / 2), int(IMAGE_SIZE - (center[0, 1] + EYE_H / 2))),
            (int(center[0, 0] - EYE_W / 2), int(IMAGE_SIZE - (center[0, 0] + EYE_W / 2)))
            ))
        self.pad_er = PadOnes((
            (int(center[1, 1] - EYE_H / 2), int(IMAGE_SIZE - (center[1, 1] + EYE_H / 2))),
            (int(center[1, 0] - EYE_W / 2), int(IMAGE_SIZE - (center[1, 0] + EYE_W / 2)))
            ))
        self.pad_no = PadOnes((
            (int(center[2, 1] - NOSE_H / 2), int(IMAGE_SIZE - (center[2, 1] + NOSE_H / 2))),
            (int(center[2, 0] - NOSE_W / 2), int(IMAGE_SIZE - (center[2, 0] + NOSE_W / 2)))
            ))
        self.pad_mo = PadOnes((
            (int(center[3, 1] - MOUTH_H / 2), int(IMAGE_SIZE - (center[3, 1] + MOUTH_H / 2))),
            (int(center[3, 0] - MOUTH_W / 2), int(IMAGE_SIZE - (center[3, 0] + MOUTH_W / 2)))
            ))

    def construct(self, real_A, real_A_bg, real_A_eyel, real_A_eyer,
                  real_A_nose, real_A_mouth, real_A_hair,
                  mask, mask2):
        return self._generate_output(real_A, real_A_bg, real_A_eyel,
                                     real_A_eyer, real_A_nose, real_A_mouth,
                                     real_A_hair, mask, mask2)
