# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
import mindspore.ops as ops
from src.mix_transformer import MitB0, MitB1, MitB2, MitB3, MitB4, MitB5
from src.segformer_head import SegFormerHead
from src.model_utils.common import VERSION_GT_2_0_0


class SegFormer(nn.Cell):
    def __init__(self, backbone_name, num_classes, sync_bn=False):
        super(SegFormer, self).__init__()
        if backbone_name == 'mit_b0':
            self.backbone = MitB0()
        elif backbone_name == 'mit_b1':
            self.backbone = MitB1()
        elif backbone_name == 'mit_b2':
            self.backbone = MitB2()
        elif backbone_name == 'mit_b3':
            self.backbone = MitB3()
        elif backbone_name == 'mit_b4':
            self.backbone = MitB4()
        elif backbone_name == 'mit_b5':
            self.backbone = MitB5()
        else:
            raise KeyError(f"Unsupported backbone {backbone_name}")

        self.decode_head = SegFormerHead(self.backbone.embed_dims, self.backbone.embed_dims[-1],
                                         num_classes=num_classes, sync_bn=sync_bn)
        self.version_gt_2_0_0 = VERSION_GT_2_0_0

    def construct(self, x):
        output_resize = x.shape[2:]
        x = self.backbone(x)
        x = self.decode_head(x)
        if self.version_gt_2_0_0:
            x = ops.interpolate(x, size=output_resize, mode='bilinear', align_corners=False)
        else:
            x = ops.interpolate(x, sizes=output_resize, mode='bilinear', coordinate_transformation_mode='half_pixel')
        return x
