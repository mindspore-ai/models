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
"""PreHead"""
from itertools import product
from math import sqrt
import mindspore.nn as nn
import mindspore
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from src.config import yolact_plus_resnet50_config as cfg
from src.yolact.layers.backbone_dcnV2 import construct_backbone
from src.yolact.layers.fpn import ResNetV1Fpn
from src.yolact.utils.functions import make_net
from src.yolact.layers.functions.detection import Detectx
from src.yolact.layers.protonet import protonet
class PreHead(nn.Cell):
    """Forward processing"""
    def __init__(self, in_channels, out_channels, num_priors, num_classes, mask_dim, aspect_ratios=None, scales=None):
        super().__init__()
        self.num_priors = num_priors
        self.num_classes = num_classes
        self.mask_dim = mask_dim
        self.upfeature, out_channels = make_net(in_channels, cfg['extra_head_net'])
        self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **cfg['head_layer_params'])
        self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg['head_layer_params'])
        self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **cfg['head_layer_params'])

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.sigmoid = P.Sigmoid()
        self.relu = P.ReLU()
        self.tanh = P.Tanh()
        self.zeros = P.Zeros()
        self.cast = P.Cast()

        self.num_classes = cfg['num_classes']
        self.mask_dim = cfg['mask_dim']  # Defined by Yolact

        self.aspect_ratios = aspect_ratios if not aspect_ratios is None else [[[1]]]

        self.scales = scales if not scales is None else [1]
        self.num_priors = sum(len(x) * len(self.scales) for x in self.aspect_ratios)

        self.last_conv_size = None
        self.last_img_size = None
        self.concat = P.Concat()
        prior_data = []

        idx = 0
        for k in [69, 35, 18, 9, 5]:
            # Iteration order is important (it has to sync up with the convout)
            for j, i in product(range(k), range(k)):
                # +0.5 because priors are in center-size notation
                x = (i + 0.5) / k  # conv_w
                y = (j + 0.5) / k  # conv_h

                for ars in self.aspect_ratios[idx]:
                    for scale in self.scales[idx]:
                        for ar in ars:
                            ar = sqrt(ar)
                            w = scale * ar / cfg['max_size']
                            h = scale / ar / cfg['max_size']
                            prior_data += [x, y, w, h]
            idx += 1
        prior = self.reshape(mindspore.Tensor(prior_data), (-1, 4))
        self.priors = prior

    def construct(self, x):
        """Forward"""
        x_ = self.upfeature(x)
        perm = (0, 2, 3, 1)
        x_bbox = self.transpose(self.bbox_layer(x_), perm)  # 1 69 69 12
        bbox = self.reshape(x_bbox, (x.shape[0], -1, 4))  # 1 14283 4
        x_conf = self.transpose(self.conf_layer(x_), perm)
        conf = self.reshape(x_conf, (x.shape[0], -1, self.num_classes))  # 1 14283 81
        x_mask = self.transpose(self.mask_layer(x_), perm)
        mask = self.reshape(x_mask, (x.shape[0], -1, self.mask_dim))
        mask = self.tanh(mask)
        preds = {'loc': bbox, 'conf': conf, 'mask': mask}

        return preds

class Yolact(nn.Cell):
    """Yolact"""
    def __init__(self):
        super().__init__()
        self.backbone = construct_backbone(cfg['backbone'])
        self.fpn = ResNetV1Fpn()
        self.proto_net = protonet()
        self.cast = P.Cast()
        self.num_grids = 0
        self.proto_src = cfg['mask_proto_src']
        if self.proto_src is None:
            in_channels = 3
        elif cfg['fpn'] is not None:
            in_channels = cfg['fpn']['num_features']
        else:
            in_channels = self.backbone.channels[self.proto_src]
        in_channels += self.num_grids

        cfg['mask_dim'] = 32

        if cfg['mask_proto_bias']:
            cfg['mask_dim'] += 1

        self.selected_layers = cfg['backbone']['selected_layers']
        self.selected_layers = list(range(len(self.selected_layers) + cfg['fpn']['num_downsample']))
        src_channels = [cfg['fpn']['num_features']] * len(self.selected_layers)

        self.num_classes = cfg['num_classes']
        self.mask_dim = cfg['mask_dim']  # Defined by Yolact
        self.num_priors = sum(len(x) * len(cfg['backbone']['pred_scales'][0])
                              for x in cfg['backbone']['pred_aspect_ratios'][0])
        self.prehead = PreHead(in_channels=cfg['fpn']['num_features'], out_channels=cfg['fpn']['num_features'],
                               num_priors=self.num_priors, num_classes=self.num_classes, mask_dim=self.mask_dim,
                               aspect_ratios=cfg['backbone']['pred_aspect_ratios'],
                               scales=cfg['backbone']['pred_scales'])
        self.priors = self.prehead.priors
        self.concat = P.Concat(1)
        self.relu = P.ReLU()
        self.transpose = P.Transpose()

        if cfg['use_semantic_segmentation_loss']:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg['num_classes'] - 1,
                                               kernel_size=1, pad_mode='pad', has_bias=True)

        # For use in evaluation
        self.softmax = P.Softmax()
        self.detect = Detectx(cfg['num_classes'], bkg_label=0, top_k=cfg['nms_top_k'],
                              conf_thresh=cfg['nms_conf_thresh'], nms_thresh=cfg['nms_thresh'])


    def construct(self, img_data=None):
        """Forward"""
        outs = self.backbone(img_data)
        outs_fpn = self.fpn(outs)
        proto_x = outs_fpn[self.proto_src]   # 1 256 69 69
        proto_out = self.proto_net(proto_x)
        pred_outs = {'loc': [], 'conf': [], 'mask': [], 'priors': [], 'proto': []}
        pred_loc = ()
        pred_conf = ()
        pred_mask = ()

        for i in range(5):
            idx = self.selected_layers[i]
            pred_x = outs_fpn[idx]  # A tensor
            p = self.prehead(pred_x)  # Pass in a tensor to generate a dictionary

            tmp_loc = p['loc']
            tmp_conf = p['conf']
            tmp_mask = p['mask']

            pred_loc += (tmp_loc,)
            pred_conf += (tmp_conf,)
            pred_mask += (tmp_mask,)

        pred_outs['loc'] = self.concat(pred_loc) # 1 19248 4
        pred_outs['conf'] = self.concat(pred_conf)  # 1 19248 81
        pred_outs['mask'] = self.concat(pred_mask) # 1 19248 32
        pred_outs['priors'] = self.priors    # 19248 4
        pred_outs['priors'] = F.stop_gradient(pred_outs['priors'])
        pred_outs['proto'] = proto_out         # 1 138 138 32

        if self.training:
            pred_outs['segm'] = self.semantic_seg_conv(outs_fpn[0])
            return pred_outs

        pred_outs['conf'] = self.softmax(pred_outs['conf'])
        return self.detect(pred_outs, self)
