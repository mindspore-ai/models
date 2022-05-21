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
"""YOLOv3 based on DarkNet."""
import math

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn
from mindspore import ops
from mindspore.ops import constexpr
from mindspore.ops import operations as P

from cfg.config import config as default_config
from src.utils import DecodeDeltaMap
from src.utils import SoftmaxCE
from src.utils import create_anchors_vec


def _conv_bn_relu(
        in_channel,
        out_channel,
        ksize,
        stride=1,
        padding=0,
        dilation=1,
        alpha=0.1,
        momentum=0.9,
        eps=1e-5,
        pad_mode="same",
):
    """
    Set a conv2d, BN and relu layer.
    """
    dbl = nn.SequentialCell(
        [
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                pad_mode=pad_mode,
            ),
            nn.BatchNorm2d(out_channel, momentum=momentum, eps=eps),
            nn.LeakyReLU(alpha),
        ]
    )

    return dbl


@constexpr
def batch_index(batch_size):
    """
    Construct index for each image in batch.

    Example:
        if batch_size = 2, returns ms.Tensor([[0], [1]])
    """
    batch_i = ms.Tensor(msnp.arange(batch_size).reshape(-1, 1), dtype=ms.int32)

    return batch_i


class YoloBlock(nn.Cell):
    """
    YoloBlock for YOLOv3.

    Args:
        in_channels (int): Input channel.
        out_chls (int): Middle channel.
        out_channels (int): Output channel.
        config (class): Config with model and training params.

    Returns:
        c5 (ms.Tensor): Feature map to feed at next layers.
        out (ms.Tensor): Output feature map.
        emb (ms.Tensor): Output embeddings.

    Examples:
        YoloBlock(1024, 512, 24)
    """

    def __init__(
            self,
            in_channels,
            out_chls,
            out_channels,
            config=default_config,
    ):
        super().__init__()
        out_chls_2 = out_chls * 2

        emb_dim = config.embedding_dim

        self.conv0 = _conv_bn_relu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv2 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv4 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, has_bias=True)

        self.emb_conv = nn.Conv2d(out_chls, emb_dim, kernel_size=3, stride=1, has_bias=True)

    def construct(self, x):
        """
        Feed forward feature map to YOLOv3 block
        to get detections and embeddings.
        """
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        emb = self.emb_conv(c5)

        out = self.conv6(c6)

        return c5, out, emb


class YOLOv3(nn.Cell):
    """
    YOLOv3 Network.

    Note:
        backbone = darknet53

    Args:
        backbone_shape (list): Darknet output channels shape.
        backbone (nn.Cell): Backbone Network.
        out_channel (int): Output channel.

    Returns:
       small_feature (ms.Tensor): Feature_map with shape (batch_size, backbone_shape[2], h/8, w/8).
       medium_feature (ms.Tensor): Feature_map with shape (batch_size, backbone_shape[3], h/16, w/16).
       big_feature (ms.Tensor): Feature_map with shape (batch_size, backbone_shape[4], h/32, w/32).

    Examples:
        YOLOv3(
            backbone_shape=[64, 128, 256, 512, 1024]
            backbone=darknet53(),
            out_channel=24,
            )
    """

    def __init__(self, backbone_shape, backbone, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.backbone = backbone
        self.backblock0 = YoloBlock(
            in_channels=backbone_shape[-1],  # 1024
            out_chls=backbone_shape[-2],  # 512
            out_channels=out_channel,  # 24
        )

        self.conv1 = _conv_bn_relu(
            in_channel=backbone_shape[-2],  # 1024
            out_channel=backbone_shape[-2] // 2,  # 512
            ksize=1,
        )
        self.backblock1 = YoloBlock(
            in_channels=backbone_shape[-2] + backbone_shape[-3],  # 768
            out_chls=backbone_shape[-3],  # 256
            out_channels=out_channel,  # 24
        )

        self.conv2 = _conv_bn_relu(
            in_channel=backbone_shape[-3],  # 256
            out_channel=backbone_shape[-3] // 2,  # 128
            ksize=1,
        )
        self.backblock2 = YoloBlock(
            in_channels=backbone_shape[-3] + backbone_shape[-4],  # 384
            out_chls=backbone_shape[-4],  # 128
            out_channels=out_channel,  # 24
        )
        self.concat = P.Concat(axis=1)

        self.freeze_bn()

    def freeze_bn(self):
        """Freeze batch norms."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                cell.beta.requires_grad = False
                cell.gamma.requires_grad = False

    def construct(self, x):
        """
        Feed forward image to FPN to get
        3 feature maps from different scales.
        """
        # input_shape of x is (batch_size, 3, h, w)
        img_hight = P.Shape()(x)[2]
        img_width = P.Shape()(x)[3]
        feature_map1, feature_map2, feature_map3 = self.backbone(x)
        con1, small_object_output, sml_emb = self.backblock0(feature_map3)

        con1 = self.conv1(con1)
        ups1 = P.ResizeNearestNeighbor((img_hight // 16, img_width // 16))(con1)
        con1 = self.concat((ups1, feature_map2))
        con2, medium_object_output, med_emb = self.backblock1(con1)

        con2 = self.conv2(con2)
        ups2 = P.ResizeNearestNeighbor((img_hight // 8, img_width // 8))(con2)
        con3 = self.concat((ups2, feature_map1))
        _, big_object_output, big_emb = self.backblock2(con3)

        small_feature = self.concat((small_object_output, sml_emb))
        medium_feature = self.concat((medium_object_output, med_emb))
        big_feature = self.concat((big_object_output, big_emb))

        return small_feature, medium_feature, big_feature


class YOLOLayer(nn.Cell):
    """
    Head for loss calculation of classification confidence,
    bbox regression and ids embedding learning .

    Args:
        anchors (list): Absolute sizes of anchors (w, h).
        nid (int): Number of identities in whole train datasets.
        emb_dim (int): Size of embedding.
        nc (int): Number of ground truth classes.

    Returns:
        loss (ms.Tensor): Auto balanced loss, calculated from conf, bbox and ids.
    """

    def __init__(
            self,
            anchors,
            nid,
            emb_dim,
            nc=default_config.num_classes,
    ):
        super().__init__()
        self.anchors = ms.Tensor(anchors, ms.float32)
        self.na = len(anchors)  # Number of anchors (4)
        self.nc = nc  # Number of classes (1)
        self.nid = nid  # Number of identities
        self.emb_dim = emb_dim

        # Set necessary operations and constants
        self.normalize = ops.L2Normalize(axis=1, epsilon=1e-12)
        self.argmax = ops.ArgMaxWithValue(axis=1)
        self.expand_dims = ops.ExpandDims()
        self.reduce_sum = ops.ReduceSum()
        self.fill = ops.Fill()
        self.exp = ops.Exp()
        self.zero_tensor = ms.Tensor([0])

        # Set eps to escape division by zero
        self.eps = ms.Tensor(1e-16, dtype=ms.float32)

        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.softmax_loss = SoftmaxCE()
        self.id_loss = SoftmaxCE()

        # Set trainable parameters for loss computation
        self.s_c = ms.Parameter(-4.15 * ms.Tensor([1]))  # -4.15
        self.s_r = ms.Parameter(-4.85 * ms.Tensor([1]))  # -4.85
        self.s_id = ms.Parameter(-2.3 * ms.Tensor([1]))  # -2.3

        self.emb_scale = math.sqrt(2) * math.log(self.nid - 1)

    def construct(self, p_cat, tconf, tbox, tids, emb_indices, classifier):
        """
        Feed forward output from the FPN,
        calculate confidence loss, bbox regression loss, target id loss,
        apply auto-balancing loss strategy.
        """
        # Get detections and embeddings from model concatenated output.
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nb, ngh, ngw = p.shape[0], p.shape[-2], p.shape[-1]

        p = p.view(nb, self.na, self.nc + 5, ngh, ngw).transpose(0, 1, 3, 4, 2)  # prediction
        p_emb = p_emb.transpose(0, 2, 3, 1)
        p_box = p[..., :4]
        p_conf = p[..., 4:6].transpose(0, 4, 1, 2, 3)

        mask = (tconf > 0).astype('float32')

        # Compute losses
        nm = self.reduce_sum(mask)  # number of anchors (assigned to targets)
        p_box = p_box * self.expand_dims(mask, -1)
        tbox = tbox * self.expand_dims(mask, -1)
        lbox = self.smooth_l1_loss(p_box, tbox)
        lbox = lbox * self.expand_dims(mask, -1)
        lbox = self.reduce_sum(lbox) / (nm * 4 + self.eps)

        lconf = self.softmax_loss(p_conf.transpose(0, 2, 3, 4, 1), tconf, ignore_index=-1)

        # Construct indices for selecting embeddings
        # from the flattened view of the model output
        # (corresponding to the embeddings prediction).
        #
        # Set flattened mask to existing detections
        # and apply it to flattened indices to nullify if it is no detection.
        emb_indices_batch_stride = emb_indices + batch_index(nb) * ngh * ngw  # Shape (nb, k_max)
        emb_indices_mask_flat = (emb_indices.reshape(-1) > 0).astype('float32')  # Shape (nb x k_max)
        emb_indices_flat = (emb_indices_batch_stride.reshape(-1) * emb_indices_mask_flat).astype('int32')

        # Flatten embs and take which is associate to flattened emb index
        emb_flat = p_emb.view(-1, self.emb_dim)  # Shape (nb x ngh x ngw, emb_dim)
        embedding = emb_flat[emb_indices_flat]  # Shape (nb x k_max, emb_dim)
        embedding = self.emb_scale * self.normalize(embedding)

        # Flatten max tids and take according to index
        _, tids = self.argmax(tids.astype('float32'))  # Shape (nb, ngh, ngw)
        tids_flat = tids.view(-1)[emb_indices_flat]  # Shape (nb x k_max)

        # Apply flattened emb mask for nullify if it is no detections
        # and subtract 1 where no detection to apply ignore mask into loss calculation.
        tids_flat_masked = tids_flat * emb_indices_mask_flat
        tids_flat_with_ignore = tids_flat_masked + (emb_indices_mask_flat - 1)

        # Apply FC layer to embeddings
        # and compute loss by custom loss with ignore index = -1.
        logits = classifier(embedding)
        lid = self.id_loss(logits, tids_flat_with_ignore.astype('int32'), ignore_index=-1)

        # Apply auto-balancing loss strategy
        loss = self.exp((-1) * self.s_r) * lbox + \
               self.exp((-1) * self.s_c) * lconf + \
               self.exp((-1) * self.s_id) * lid + \
               (self.s_r + self.s_c + self.s_id)
        loss *= 0.5

        return loss.squeeze()


class JDE(nn.Cell):
    """
    JDE Network.

    Args:
        extractor (nn.Cell): Backbone, which extracts feature maps.
        config (class): Config with model and training params.
        nid (int): Number of identities in whole train datasets.
        ne (int): Size of embedding.

    Returns:
        loss (ms.Tensor): Sum of 3 losses from each head.

    Note:
        backbone = YOLOv3 with darknet53
        head = 3 similar heads for each feature map size
    """

    def __init__(self, extractor, config, nid, ne):
        super().__init__()
        anchors = config.anchor_scales
        anchors1 = anchors[0:4]
        anchors2 = anchors[4:8]
        anchors3 = anchors[8:12]

        self.backbone = extractor

        # Set loss cell layers for different scales
        self.head_s = YOLOLayer(anchors3, nid, ne)
        self.head_m = YOLOLayer(anchors2, nid, ne)
        self.head_b = YOLOLayer(anchors1, nid, ne)

        # Set classifier for embeddings
        self.classifier = nn.Dense(ne, nid)

    def construct(
            self,
            images,
            tconf_s,
            tbox_s,
            tid_s,
            tconf_m,
            tbox_m,
            tid_m,
            tconf_b,
            tbox_b,
            tid_b,
            mask_s,
            mask_m,
            mask_b,
    ):
        """
        Feed forward image to FPN, get 3 feature maps with different sizes,
        put it into 3 heads, corresponding to size,
        get auto-balanced losses, summarize them.
        """
        # Apply FPN to image to get 3 feature map with different scales
        small, medium, big = self.backbone(images)

        # Calculate losses for each feature map
        out_s = self.head_s(small, tconf_s, tbox_s, tid_s, mask_s, self.classifier)
        out_m = self.head_m(medium, tconf_m, tbox_m, tid_m, mask_m, self.classifier)
        out_b = self.head_b(big, tconf_b, tbox_b, tid_b, mask_b, self.classifier)

        loss = (out_s + out_m + out_b) / 3

        return loss


class YOLOLayerEval(nn.Cell):
    """
    Head for detection and tracking.

    Args:
        anchor (list): Absolute sizes of anchors (w, h).
        nc (int): Number of ground truth classes.

    Returns:
        prediction (ms.Tensor): Model predictions for confidences, boxes and embeddings.
    """

    def __init__(
            self,
            anchor,
            stride,
            nc=default_config.num_classes,
    ):
        super().__init__()
        self.na = len(anchor)  # number of anchors (4)
        self.nc = nc  # number of classes (1)
        self.anchor_vec = anchor
        self.stride = stride

        self.argmax = ops.ArgMaxWithValue(axis=1)
        self.expand_dims = ops.ExpandDims()
        self.softmax = nn.Softmax(axis=1)
        self.normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)
        self.tile = ops.Tile()
        self.fill = ops.Fill()
        self.concat = ops.Concat(axis=-1)

        self.decode_map = DecodeDeltaMap()

    def construct(self, p_cat):
        """
        Feed forward output from the FPN,
        calculate prediction corresponding to anchor.
        """
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nb, ngh, ngw = p.shape[0], p.shape[-2], p.shape[-1]

        p = p.view(nb, self.na, self.nc + 5, ngh, ngw).transpose(0, 1, 3, 4, 2)  # prediction
        p_emb = p_emb.transpose(0, 2, 3, 1)
        p_box = p[..., :4]
        p_conf = p[..., 4:6].transpose(0, 4, 1, 2, 3)  # conf
        p_conf = self.expand_dims(self.softmax(p_conf)[:, 1, ...], -1)
        p_emb = self.normalize(self.tile(self.expand_dims(p_emb, 1), (1, self.na, 1, 1, 1)))

        p_cls = self.fill(ms.float32, (nb, self.na, ngh, ngw, 1), 0)  # temp
        p = self.concat((p_box, p_conf, p_cls, p_emb))

        # Decode bbox delta to the absolute cords
        p_1 = self.decode_map(p[..., :4], self.anchor_vec)
        p_1 = p_1 * self.stride

        p = self.concat((p_1.astype('float32'), p[..., 4:]))
        prediction = p.reshape(nb, -1, p.shape[-1])

        return prediction


class JDEeval(nn.Cell):
    """
     JDE Network.

     Note:
         backbone = YOLOv3 with darknet53.
         head = 3 similar heads for each feature map size.

     Returns:
         output (ms.Tensor): Tensor with concatenated outputs from each head.
         output_top_k (ms.Tensor): Output tensor of top_k best proposals by confidence.

    """

    def __init__(self, extractor, config):
        super().__init__()
        anchors, strides = create_anchors_vec(config.anchor_scales)
        anchors = ms.Tensor(anchors, dtype=ms.float32)
        strides = ms.Tensor(strides, dtype=ms.float32)

        self.backbone = extractor

        self.head_s = YOLOLayerEval(anchors[0], strides[0])
        self.head_m = YOLOLayerEval(anchors[1], strides[1])
        self.head_b = YOLOLayerEval(anchors[2], strides[2])

        self.concatenate = ops.Concat(axis=1)
        self.top_k = ops.TopK(sorted=False)
        self.k = 800

    def construct(self, images):
        """
        Feed forward image to FPN, get 3 feature maps with different sizes,
        put them into 3 heads, corresponding to size,
        get concatenated output of proposals.
        """
        small, medium, big = self.backbone(images)

        out_s = self.head_s(small)
        out_m = self.head_m(medium)
        out_b = self.head_b(big)

        output = self.concatenate((out_s, out_m, out_b))

        _, top_k_indices = self.top_k(output[:, :, 4], self.k)
        output_top_k = output[0][top_k_indices]

        return output, output_top_k
