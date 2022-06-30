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

import os
import mindspore
import mindspore.numpy as msnp
from mindspore import nn, Tensor, ops, load_checkpoint, Parameter
import numpy as np
from models.upsample import Upsample
from models.resnet import resnet50
from models.layer import ConvBottleNeck, HgNet
from models.uv_generator import Index_UV_Generator
from utils.objfile import read_obj

# Warp elements in image space to UV space.
def warp_feature(dp_out, feature_map, uv_res):
    """
    C: channel number of the input feature map;  H: height;  W: width

    :param dp_out: IUV image in shape (batch_size, 3, H, W)
    :param feature_map: Local feature map in shape (batch_size, C, H, W)
    :param uv_res: The resolution of the transferred feature map in UV space.

    :return: warped_feature: Feature map in UV space with shape (batch_size, C+3, uv_res, uv_res)
    The x, y coordinates in the image space and mask will be added as the last 3 channels
     of the warped feature, so the channel number of warped feature is C+3.
    """

    expand_dims = ops.ExpandDims()
    dp_mask = expand_dims(dp_out[:, 0], 1)        # I channel, confidence of being foreground
    dp_uv = dp_out[:, 1:]                         # UV channels, UV coordinates
    thre = 0.5                                    # The threshold of foreground and background.
    B, C, H, W = feature_map.shape

    # Get the sampling index of every pixel in batch_size dimension.
    index_batch = msnp.arange(0, B, dtype=mindspore.int64)[:, None, None].repeat(H, 1).repeat(W, 2)
    index_batch = index_batch.view(-1).astype("int64")

    # Get the sampling index of every pixel in H and W dimension.
    tmp_x = msnp.arange(0, W, dtype=mindspore.int64)
    tmp_y = msnp.arange(0, H, dtype=mindspore.int64)

    meshgrid = ops.Meshgrid(indexing="ij")
    y, x = meshgrid((tmp_y, tmp_x))

    y = ops.Tile()(y.view(-1), (1, B))[0]
    x = ops.Tile()(x.view(-1), (1, B))[0]

    # Sample the confidence of every pixel,
    # and only preserve the pixels belong to foreground.
    conf = dp_mask[index_batch, 0, y, x]
    valid = conf > thre
    ind = valid.astype('int64')

    warped_feature = Tensor(msnp.zeros((B, uv_res, uv_res, C + 3))).transpose(0, 3, 1, 2)
    if ind.sum() == 0:
        warped_feature = warped_feature

    elif ind.sum() != 0:
        index_batch = mindspore.Tensor(index_batch.asnumpy())
        y = mindspore.Tensor(y.asnumpy())
        x = mindspore.Tensor(x.asnumpy())
        uv = dp_uv[index_batch, :, y, x]
        num_pixel = uv.shape[0]
        # Get the corresponding location in UV space
        uv = uv * (uv_res - 1)
        m_round = ops.Round()
        uv_round = m_round(uv).astype("int64").clip(xmin=0, xmax=uv_res-1)

        # We first process the transferred feature in shape (batch_size * H * W, C+3),
        # so we need to get the location of each pixel in the two-dimension feature vector.
        index_uv = (uv_round[:, 1] * uv_res + uv_round[:, 0]).copy() + index_batch * uv_res * uv_res

        # Sample the feature of foreground pixels
        sampled_feature = feature_map[index_batch, :, y, x]
        # Scale x,y coordinates to [-1, 1] and
        # concatenated to the end of sampled feature as extra channels.
        y = (2 * y.astype("float32") / (H - 1)) - 1
        x = (2 * x.astype("float32") / (W - 1)) - 1
        concat = ops.Concat(-1)
        sampled_feature = concat([sampled_feature, x[:, None], y[:, None]])

        zeros = ops.Zeros()
        ones = ops.Ones()
        # Multiple pixels in image space may be transferred to the same location in the UV space.
        # warped_w is used to record the number of the pixels transferred to every location.
        warped_w = zeros((B * uv_res * uv_res, 1), sampled_feature.dtype)
        index_add = ops.IndexAdd(axis=0)
        warped_w = index_add(warped_w, index_uv.astype("int32"), ones((num_pixel, 1), sampled_feature.dtype))

        # Transfer the sampled feature to UV space.
        # Feature vectors transferred to the sample location will be accumulated.
        warped_feature = zeros((B * uv_res * uv_res, C + 2), sampled_feature.dtype)
        warped_feature = index_add(warped_feature, index_uv.astype("int32"), sampled_feature)

        # Normalize the accumulated feature with the pixel number.
        warped_feature = warped_feature/(warped_w + 1e-8)
        # Concatenate the mask channel at the end.
        warped_feature = concat([warped_feature, (warped_w > 0).astype("float32")])
        # Reshape the shape to (batch_size, C+3, uv_res, uv_res)
        warped_feature = warped_feature.reshape(B, uv_res, uv_res, C + 3).transpose(0, 3, 1, 2)
    return warped_feature

# DPNet:returns densepose result
class DPNet(nn.Cell):
    def __init__(self, warp_lv=2, norm_type='BN'):
        super(DPNet, self).__init__()
        nl_layer = nn.ReLU()
        self.warp_lv = warp_lv
        # image encoder
        self.resnet = resnet50(pretrained=True)
        # dense pose line
        dp_layers = []
        # correspond res[224, 112,  56,  28,   14,    7]
        channel_list = [3, 64, 256, 512, 1024, 2048]
        for i in range(warp_lv, 5):
            in_channels = channel_list[i + 1]
            out_channels = channel_list[i]

            dp_layers.append(
                nn.SequentialCell(
                    Upsample(),
                    ConvBottleNeck(in_channels=in_channels, out_channels=out_channels,
                                   nl_layer=nl_layer, norm_type=norm_type)
                    )
                )

        self.dp_layers = nn.CellList(dp_layers)
        self.dp_uv_end = nn.SequentialCell(ConvBottleNeck(channel_list[warp_lv], 32, nl_layer, norm_type=norm_type),
                                           nn.Conv2d(32, 2, kernel_size=1, has_bias=True, bias_init='normal'),
                                           nn.Sigmoid())

        self.dp_mask_end = nn.SequentialCell(ConvBottleNeck(channel_list[warp_lv], 32, nl_layer, norm_type=norm_type),
                                             nn.Conv2d(32, 1, kernel_size=1, has_bias=True, bias_init='normal'),
                                             nn.Sigmoid())

    def construct(self, image, UV=None):
        codes, features = self.resnet(image)
        # output densepose results
        dp_feature = features[-1]
        for i in range(len(self.dp_layers) - 1, -1, -1):
            dp_feature = self.dp_layers[i](dp_feature)
            dp_feature = dp_feature + features[i - 1 + len(features) - len(self.dp_layers)]
        dp_uv = self.dp_uv_end(dp_feature)
        dp_mask = self.dp_mask_end(dp_feature)
        ops_cat = ops.Concat(1)
        dp_out = ops_cat((dp_mask, dp_uv))

        return dp_out, dp_feature, codes

def Pretrained_DPNet(warp_level, norm_type, pretrained=False):

    model_file = "ckpt/rank0/CNet_5.ckpt"
    model = DPNet(warp_lv=warp_level, norm_type=norm_type)

    if pretrained:
        load_checkpoint(model_file, net=model)
    return model

def get_LNet(options):
    if options.model == 'DecoMR':
        uv_net = UVNet(uv_channels=options.uv_channels,
                       uv_res=options.uv_res,
                       warp_lv=options.warp_level,
                       uv_type=options.uv_type,
                       norm_type=options.norm_type)
    return uv_net

# UVNet returns location map
class UVNet(nn.Cell):
    def __init__(self, uv_channels=64, uv_res=128, warp_lv=2, uv_type='SMPL', norm_type='BN'):
        super(UVNet, self).__init__()

        nl_layer = nn.ReLU()
        self.fc_head = nn.SequentialCell(
            nn.Dense(2048, 512),
            nn.BatchNorm1d(512),
            nl_layer,
            nn.Dense(512, 256)
        )
        self.camera = nn.SequentialCell(
            nn.Dense(2048, 512),
            nn.BatchNorm1d(512),
            nl_layer,
            nn.Dense(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dense(256, 3)
            )

        self.warp_lv = warp_lv
        channel_list = [3, 64, 256, 512, 1024, 2048]
        warp_channel = channel_list[warp_lv]
        self.uv_res = uv_res
        self.warp_res = int(256 // (2**self.warp_lv))

        if uv_type == "SMPL":
            ref_file = 'data/SMPL_ref_map_{0}.npy'.format(self.warp_res)
        elif uv_type == 'BF':
            ref_file = 'data/BF_ref_map_{0}.npy'.format(self.warp_res)
        if not os.path.exists(ref_file):
            sampler = Index_UV_Generator(UV_height=self.warp_res, uv_type=uv_type)
            ref_vert, _ = read_obj('data/reference_mesh.obj')
            ref_map = sampler.get_UV_map(Tensor.from_numpy(ref_vert).astype("float32"))
            np.save(ref_file, ref_map.asnumpy())
        self.ref_map = Parameter(Tensor.from_numpy(np.load(ref_file)).astype("float32").transpose(0, 3, 1, 2),
                                 name="ref_map", requires_grad=False)

        self.uv_conv1 = nn.SequentialCell(
            nn.Conv2d(256 + warp_channel + 3 + 3, 2 * warp_channel, kernel_size=1, has_bias=True, bias_init='normal'),
            nl_layer,
            nn.Conv2d(2 * warp_channel, 2 * warp_channel, kernel_size=1, has_bias=True, bias_init='normal'),
            nl_layer,
            nn.Conv2d(2 * warp_channel, warp_channel, kernel_size=1, has_bias=True, bias_init='normal'))

        uv_lv = 0 if uv_res == 256 else 1
        self.hg = HgNet(in_channels=warp_channel, level=5-warp_lv, nl_layer=nl_layer, norm_type=norm_type)

        cur = min(8, 2 ** (warp_lv - uv_lv))
        prev = cur
        self.uv_conv2 = ConvBottleNeck(warp_channel, uv_channels * cur, nl_layer, norm_type=norm_type)

        layers = []
        for lv in range(warp_lv, uv_lv, -1):
            cur = min(prev, 2 ** (lv - uv_lv - 1))
            layers.append(
                nn.SequentialCell(Upsample(),
                                  ConvBottleNeck(uv_channels * prev, uv_channels * cur, nl_layer, norm_type=norm_type))
            )
            prev = cur
        self.decoder = nn.SequentialCell(layers)
        self.uv_end = nn.SequentialCell(ConvBottleNeck(uv_channels, 32, nl_layer, norm_type=norm_type),
                                        nn.Conv2d(32, 3, kernel_size=1, has_bias=True, bias_init='normal'))

    def construct(self, dp_out, dp_feature, codes):
        n_batch = dp_out.shape[0]
        local_feature = warp_feature(dp_out, dp_feature, self.warp_res)

        global_feature = self.fc_head(codes)

        shape_1 = (-1, -1, self.warp_res, self.warp_res)
        global_feature = ops.BroadcastTo(shape_1)(global_feature[:, :, None, None])

        self.ref_map = self.ref_map.astype(local_feature.dtype)
        concat = ops.Concat(1)
        shape_2 = (n_batch, -1, -1, -1)
        uv_map = concat((local_feature, global_feature, ops.BroadcastTo(shape_2)(self.ref_map)))

        uv_map = self.uv_conv1(uv_map)
        uv_map = self.hg(uv_map)
        uv_map = self.uv_conv2(uv_map)
        uv_map = self.decoder(uv_map)
        uv_map = self.uv_end(uv_map).transpose(0, 2, 3, 1)
        cam = self.camera(codes)

        return uv_map, cam
