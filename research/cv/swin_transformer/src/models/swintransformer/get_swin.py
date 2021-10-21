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
"""Get SwinTransformer of different size for args"""
from .swin_transformer import SwinTransformer


def get_swintransformer(args):
    """get swintransformer according to args"""
    # override args
    image_size = args.image_size
    patch_size = args.patch_size
    in_chans = args.in_channel
    embed_dim = args.embed_dim
    depths = args.depths
    num_heads = args.num_heads
    window_size = args.window_size
    drop_path_rate = args.drop_path_rate
    mlp_ratio = args.mlp_ratio
    qkv_bias = True
    qk_scale = None
    ape = args.ape
    patch_norm = args.patch_norm
    print(25 * "=" + "MODEL CONFIG" + 25 * "=")
    print(f"==> IMAGE_SIZE:         {image_size}")
    print(f"==> PATCH_SIZE:         {patch_size}")
    print(f"==> NUM_CLASSES:        {args.num_classes}")
    print(f"==> EMBED_DIM:          {embed_dim}")
    print(f"==> NUM_HEADS:          {num_heads}")
    print(f"==> DEPTHS:             {depths}")
    print(f"==> WINDOW_SIZE:        {window_size}")
    print(f"==> MLP_RATIO:          {mlp_ratio}")
    print(f"==> QKV_BIAS:           {qkv_bias}")
    print(f"==> QK_SCALE:           {qk_scale}")
    print(f"==> DROP_PATH_RATE:     {drop_path_rate}")
    print(f"==> APE:                {ape}")
    print(f"==> PATCH_NORM:         {patch_norm}")
    print(25 * "=" + "FINISHED" + 25 * "=")
    model = SwinTransformer(image_size=image_size,
                            patch_size=patch_size,
                            in_chans=in_chans,
                            num_classes=args.num_classes,
                            embed_dim=embed_dim,
                            depths=depths,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=0.,
                            drop_path_rate=drop_path_rate,
                            ape=ape,
                            patch_norm=patch_norm)
    # print(model)
    return model


def swin_tiny_patch4_window7_224(args):
    """swin_tiny_patch4_window7_224"""
    return get_swintransformer(args)
