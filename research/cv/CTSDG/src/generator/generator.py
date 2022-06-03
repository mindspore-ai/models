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
"""generator"""

from mindspore import nn
from mindspore import ops

from src.generator.bigff import BiGFF
from src.generator.cfa import CFA
from src.generator.pconv import PConvBNActiv
from src.generator.projection import Feature2Structure
from src.generator.projection import Feature2Texture
from src.initializer import weights_init


class Generator(nn.Cell):
    """generator"""
    def __init__(self, image_in_channels=3, edge_in_channels=2, out_channels=3):
        super().__init__()
        # texture encoder-decoder
        self.ec_texture_layers = nn.CellList([
            PConvBNActiv(image_in_channels, 64, bn=False, sample='down-7'),
            PConvBNActiv(64, 128, sample='down-5'),
            PConvBNActiv(128, 256, sample='down-5'),
            PConvBNActiv(256, 512, sample='down-3'),
            PConvBNActiv(512, 512, sample='down-3'),
            PConvBNActiv(512, 512, sample='down-3'),
            PConvBNActiv(512, 512, sample='down-3')
        ])

        self.dc_texture_layers = nn.CellList([
            PConvBNActiv(64 + out_channels, 64, activation='leaky'),
            PConvBNActiv(128 + 64, 64, activation='leaky'),
            PConvBNActiv(256 + 128, 128, activation='leaky'),
            PConvBNActiv(512 + 256, 256, activation='leaky'),
            PConvBNActiv(512 + 512, 512, activation='leaky'),
            PConvBNActiv(512 + 512, 512, activation='leaky'),
            PConvBNActiv(512 + 512, 512, activation='leaky'),
        ])

        # structure encoder-decoder
        self.ec_structure_layers = nn.CellList([
            PConvBNActiv(edge_in_channels, 64, bn=False, sample='down-7'),
            PConvBNActiv(64, 128, sample='down-5'),
            PConvBNActiv(128, 256, sample='down-5'),
            PConvBNActiv(256, 512, sample='down-3'),
            PConvBNActiv(512, 512, sample='down-3'),
            PConvBNActiv(512, 512, sample='down-3'),
            PConvBNActiv(512, 512, sample='down-3')
        ])

        self.dc_structure_layers = nn.CellList([
            PConvBNActiv(64 + 2, 64, activation='leaky'),
            PConvBNActiv(128 + 64, 64, activation='leaky'),
            PConvBNActiv(256 + 128, 128, activation='leaky'),
            PConvBNActiv(512 + 256, 256, activation='leaky'),
            PConvBNActiv(512 + 512, 512, activation='leaky'),
            PConvBNActiv(512 + 512, 512, activation='leaky'),
            PConvBNActiv(512 + 512, 512, activation='leaky'),
        ])

        # Projection Function
        self.structure_feature_projection = Feature2Structure()
        self.texture_feature_projection = Feature2Texture()

        # Bi-directional Gated Feature Fusion
        self.bigff = BiGFF(in_channels=64, out_channels=64)

        # Contextual Feature Aggregation
        self.fusion_layer1 = nn.SequentialCell(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2)
        )
        self.cfa = CFA(in_channels=64, out_channels=64)
        self.fusion_layer2 = nn.SequentialCell(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )
        self.out_layer = nn.SequentialCell(
            nn.Conv2d(64 + 64 + 64, 3, kernel_size=1, has_bias=True),
            nn.Tanh()
        )
        weights_init(self)

    @staticmethod
    def _encoder_stage(inp, mask, encoder):
        """encoder stage"""
        ec_outputs = [inp]
        ec_outputs_masks = [mask]

        for i in range(7):
            ec_out, ec_out_mask = encoder[i](ec_outputs[-1],
                                             ec_outputs_masks[-1])
            ec_outputs.append(ec_out)
            ec_outputs_masks.append(ec_out_mask)

        return ec_outputs, ec_outputs_masks

    @staticmethod
    def _decoder_stage(dc_inp, dc_inp_mask, ec_outputs, ec_outputs_masks, decoder):
        """decoder stage"""
        for i in range(6, -1, -1):
            ec_out_skip = ec_outputs[i]
            ec_out_mask_skip = ec_outputs_masks[i]
            _, _, h, w = dc_inp.shape
            dc_inp = ops.ResizeBilinear(size=(2 * h, 2 * w), align_corners=True)(dc_inp)  # align_corners = False
            dc_inp_mask = ops.ResizeNearestNeighbor(size=(2 * h, 2 * w))(dc_inp_mask)

            dc_inp = ops.Concat(axis=1)([dc_inp, ec_out_skip])
            dc_inp_mask = ops.Concat(axis=1)([dc_inp_mask, ec_out_mask_skip])

            dc_inp, dc_inp_mask = decoder[i](dc_inp, dc_inp_mask)

        return dc_inp

    def construct(self, input_image, input_edge, mask):
        """construct"""
        input_texture_mask = ops.Concat(axis=1)((mask, mask, mask))
        ec_textures, ec_textures_masks = self._encoder_stage(input_image,
                                                             input_texture_mask,
                                                             self.ec_texture_layers)

        input_structure_mask = ops.Concat(axis=1)((mask, mask))
        ec_structures, ec_structures_masks = self._encoder_stage(input_edge,
                                                                 input_structure_mask,
                                                                 self.ec_structure_layers)

        dc_textures = self._decoder_stage(ec_structures[-1],
                                          ec_structures_masks[-1],
                                          ec_textures,
                                          ec_textures_masks,
                                          self.dc_texture_layers)
        dc_structures = self._decoder_stage(ec_textures[-1],
                                            ec_textures_masks[-1],
                                            ec_structures,
                                            ec_structures_masks,
                                            self.dc_structure_layers)

        # Projection Function
        projected_image = self.texture_feature_projection(dc_textures)
        projected_edge = self.structure_feature_projection(dc_structures)

        output_bigff = self.bigff(dc_textures, dc_structures)

        output = self.fusion_layer1(output_bigff)
        output_atten = self.cfa(output, output)
        output = self.fusion_layer2(ops.Concat(axis=1)((output, output_atten)))
        _, _, h, w = output.shape
        output = ops.ResizeBilinear(size=(2 * h, 2 * w), align_corners=True)(output)  # align_corners = False
        output = self.out_layer(ops.Concat(axis=1)((output, output_bigff)))

        return output, projected_image, projected_edge
