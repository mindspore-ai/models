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
"""TextFuseNet Rcnn for segmentation network."""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.ops import ResizeBilinear as Sample
import mindspore.common.dtype as mstype
from .roi_align import ROIAlign


class FpnSeg(nn.Cell):
    """conv layers of segmentation head"""

    def __init__(self, input_channels, output_channels, num_classes, image_shape):
        super(FpnSeg, self).__init__()
        self.conv1x1_list = nn.CellList()
        for _ in range(4):
            self.conv1x1_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0,
                                               has_bias=False).to_float(mstype.float16))
        self.conv3x3_list = nn.CellList()
        for _ in range(4):
            self.conv3x3_list.append(nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=0,
                                               has_bias=False).to_float(mstype.float16))
        self.seg_pooler = ROIAlign(out_size_w=14, out_size_h=14, spatial_scale=0.125, sample_num=1, roi_align_mode=0)
        self.conv3x3_roi_list = nn.CellList()
        for _ in range(4):
            self.conv3x3_roi_list.append(nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=0,
                                                   has_bias=False).to_float(mstype.float16))

        self.conv1x1_seg_logits = nn.Conv2d(output_channels, output_channels, kernel_size=1, stride=1, padding=0,
                                            has_bias=False).to_float(mstype.float16)
        self.seg_logits = nn.Conv2d(output_channels, num_classes, kernel_size=1, stride=1, padding=0,
                                    has_bias=True).to_float(mstype.float16)

        self.image_sampler = Sample(image_shape, True)
        self.feature_sampler = Sample((96, 160), True)

    def construct(self, x, feature_level, proposal_boxes):
        """rcnn seg submodule forward"""
        feature_fuse = self.conv1x1_list[feature_level](x[feature_level])
        for i, feature in enumerate(x):
            if i != feature_level:
                feature = self.feature_sampler(feature)
                feature = self.conv1x1_list[i](feature)
                feature_fuse += feature

        for i in range(len(self.conv3x3_list)):
            feature_fuse = self.conv3x3_list[i](feature_fuse)
        global_context = self.seg_pooler(feature_fuse, proposal_boxes)
        for i in range(len(self.conv3x3_roi_list)):
            global_context = self.conv3x3_roi_list[i](global_context)

        feature_pred = self.image_sampler(feature_fuse)
        feature_pred = self.conv1x1_seg_logits(feature_pred)
        feature_pred = self.seg_logits(feature_pred)

        return feature_pred, global_context


class RcnnSeg(nn.Cell):
    """
    Rcnn for mask subnet.

    Args:
        config (dict) - Config.
        batch_size (int) - Batchsize.
        num_classes (int) - Class number.
        target_means (list) - Means for encode function. Default: (.0, .0, .0, .0]).
        target_stds (list) - Stds for encode function. Default: (0.1, 0.1, 0.2, 0.2).

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        RcnnSeg(config=config, batch_size=1, num_classes = 64, \
             target_means=(0., 0., 0., 0.), target_stds=(0.1, 0.1, 0.2, 0.2))
    """

    def __init__(self,
                 config,
                 num_classes,
                 image_shape,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(RcnnSeg, self).__init__()
        cfg = config
        self.rcnn_loss_mask_fb_weight = Tensor(np.array(cfg.rcnn_loss_mask_fb_weight).astype(np.float16))
        self.rcnn_mask_out_channels = cfg.rcnn_mask_out_channels
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_classes = num_classes
        self.in_channels = cfg.rcnn_in_channels
        self.image_shape = image_shape
        self.fpn_mask = FpnSeg(self.in_channels, self.rcnn_mask_out_channels, self.num_classes, self.image_shape)

        self.logicaland = P.LogicalAnd()
        self.loss_mask = P.SigmoidCrossEntropyWithLogits()
        self.cast = P.Cast()
        self.sum_loss = P.ReduceSum()
        self.tile = P.Tile()
        self.one_vale = Tensor(np.array(1).astype(np.int))
        self.equal = P.Equal()
        self.expandims = P.ExpandDims()

        self.mean_loss = P.ReduceMean(keep_dims=True)
        self.cast = P.Cast()
        self.squeeze = P.Squeeze(1)
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.trans = P.Transpose()
        self.logical_not = P.LogicalNot()
        self.concat = P.Concat(axis=1)
        self.reshape = P.Reshape()
        self.image_sampler = Sample((96, 160), True)

    def construct(self, featuremap, mask_fb_targets=None, mask_label=None,
                  proposal_boxes=None):
        """rcnn seg forward"""
        feature_level = 1
        x_mask_fb, global_context = self.fpn_mask(featuremap, feature_level, proposal_boxes)
        if self.training:
            label = mask_label[0]
            index_label = self.equal(label, self.one_vale)
            index_label = self.expandims(index_label, 0)
            index_label = self.expandims(index_label, 2)
            index_label = self.expandims(index_label, 3)
            index_label = self.cast(index_label, mstype.float16)
            index_label = self.tile(index_label, (1, 1, self.image_shape[0], self.image_shape[1]))

            index_label = self.cast(index_label, mstype.float16)

            mask_fb_targets = self.cast(mask_fb_targets, mstype.float16)
            mask_fb_targets = mask_fb_targets * index_label
            mask_fb_targets = self.reduce_sum(mask_fb_targets, 1)
            mask_fb_targets = self.cast(mask_fb_targets, mstype.bool_)
            mask_fb_targets = self.cast(mask_fb_targets, mstype.float16)
            mask_fb_targets = F.stop_gradient(mask_fb_targets)

            loss_mask_fb = self.loss(x_mask_fb, mask_fb_targets)
            loss_mask_fb = loss_mask_fb * 0.1
            return loss_mask_fb, global_context
        return x_mask_fb, global_context

    def loss(self, masks_fb_pred, masks_fb_targets):
        """Loss method."""
        masks_fb_targets = self.cast(masks_fb_targets, mstype.float16)

        masks_fb_targets = self.trans(masks_fb_targets, (0, 2, 3, 1))
        masks_fb_targets = self.reshape(masks_fb_targets, (-1, 1))

        masks_fb_pred = self.trans(masks_fb_pred, (0, 2, 3, 1))
        masks_fb_pred = self.reshape(masks_fb_pred, (-1, 1))

        loss_mask_fb = self.loss_mask(masks_fb_pred, masks_fb_targets)
        loss_mask_fb = self.sum_loss(loss_mask_fb, 1)

        loss_mask_fb = self.mean_loss(loss_mask_fb, 0)
        return loss_mask_fb
