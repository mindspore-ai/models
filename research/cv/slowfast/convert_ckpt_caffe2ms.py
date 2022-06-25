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
"""convert_ckpt_cf2ms"""
import pickle
import re
import numpy as np
import mindspore as ms

def convert_ckpt_cf2ms(cf_ckpt_path):
    """
    convert caffe checkpoint to mindspore checkpoint
    """
    with open(cf_ckpt_path, "rb") as f:
        caffe_ckpt = pickle.load(f, encoding="latin1")
    print("converting ckpt from caffee to mindspore...")
    ckpt_ms_ret = []
    count = 0
    for key, value in caffe_ckpt["blobs"].items():
        converted_key = convert_key_cf2pt(key)
        converted_key = convert_key_pt2ms(converted_key)
        # filter out the following case:
        # 1. key ends with 'num_batches_tracked'(actually not exist)
        # 2. non-ndarray value, only model_iter(=46000) is filtered
        if converted_key.endswith('num_batches_tracked') \
                or not isinstance(value, np.ndarray) \
                or any(prefix in converted_key for prefix in ["momentum", "lr", "model_iter", 'projection']):
            continue
        param_dict = {'name': converted_key,
                      'data': ms.Tensor(value)}
        ckpt_ms_ret.append(param_dict)
        count += 1
        print(f'{count} caffe: {key}, {value.shape} => mindspore: {converted_key}, {value.shape}')
    ms.save_checkpoint(ckpt_ms_ret, cf_ckpt_path + '.ckpt')
    print("convert_ckpt_cf2ms successfully!")


def convert_key_cf2pt(cf_key):
    """
    Convert Caffe2 key names to PyTorch key names.
    """
    pairs = [
        # ------------------------------------------------------------
        # 'nonlocal_conv3_1_theta_w' -> 's3.pathway0_nonlocal3.conv_g.weight'
        [
            r"^nonlocal_conv([0-9]+)_([0-9]+)_(.*)",
            r"s\1.pathway0_nonlocal\2_\3",
        ],
        # 'theta' -> 'conv_theta'
        [r"^(.*)_nonlocal([0-9]+)_(theta)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'g' -> 'conv_g'
        [r"^(.*)_nonlocal([0-9]+)_(g)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'phi' -> 'conv_phi'
        [r"^(.*)_nonlocal([0-9]+)_(phi)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'out' -> 'conv_out'
        [r"^(.*)_nonlocal([0-9]+)_(out)(.*)", r"\1_nonlocal\2.conv_\3\4"],
        # 'nonlocal_conv4_5_bn_s' -> 's4.pathway0_nonlocal3.bn.weight'
        [r"^(.*)_nonlocal([0-9]+)_(bn)_(.*)", r"\1_nonlocal\2.\3.\4"],
        # ------------------------------------------------------------
        # 't_pool1_subsample_bn' -> 's1_fuse.conv_f2s.bn.running_mean'
        [r"^t_pool1_subsample_bn_(.*)", r"s1_fuse.bn.\1"],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [r"^t_pool1_subsample_(.*)", r"s1_fuse.conv_f2s.\1"],
        # 't_res4_5_branch2c_bn_subsample_bn_rm' -> 's4_fuse.conv_f2s.bias'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_bn_(.*)",
            r"s\1_fuse.bn.\3",
        ],
        # 't_pool1_subsample' -> 's1_fuse.conv_f2s'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch2c_bn_subsample_(.*)",
            r"s\1_fuse.conv_f2s.\3",
        ],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway0_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^res_conv1_bn_(.*)", r"s1.pathway0_stem.bn.\1"],
        # 'conv1_xy_w_momentum' -> 's1.pathway0_stem.conv_xy.'
        [r"^conv1_xy(.*)", r"s1.pathway0_stem.conv_xy\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway0_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^res_conv1_(.*)", r"s1.pathway0_stem.conv.\1"],
        # ------------------------------------------------------------
        # 'res4_4_branch_2c_bn_b' -> 's4.pathway0_res4.branch2.c_bn_b'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)([a-z])_(.*)",
            r"s\1.pathway1_res\2.branch\3.\4_\5",
        ],
        # 'res_conv1_bn_' -> 's1.pathway0_stem.bn.'
        [r"^t_res_conv1_bn_(.*)", r"s1.pathway1_stem.bn.\1"],
        # 'conv1_w_momentum' -> 's1.pathway0_stem.conv.'
        [r"^t_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # 'res4_0_branch1_w' -> 'S4.pathway0_res0.branch1.weight'
        [
            r"^t_res([0-9]+)_([0-9]+)_branch([0-9]+)_(.*)",
            r"s\1.pathway1_res\2.branch\3_\4",
        ],
        # 'res_conv1_' -> 's1.pathway0_stem.conv.'
        [r"^t_res_conv1_(.*)", r"s1.pathway1_stem.conv.\1"],
        # ------------------------------------------------------------
        # pred_ -> head.projection.
        [r"pred_(.*)", r"head.projection.\1"],
        # '.b_bn_fc' -> '.se.fc'
        [r"(.*)b_bn_fc(.*)", r"\1se.fc\2"],
        # conv_5 -> head.conv_5.
        [r"conv_5(.*)", r"head.conv_5\1"],
        # conv_5 -> head.conv_5.
        [r"lin_5(.*)", r"head.lin_5\1"],
        # '.bn_b' -> '.weight'
        [r"(.*)bn.b\Z", r"\1bn.bias"],
        # '.bn_s' -> '.weight'
        [r"(.*)bn.s\Z", r"\1bn.weight"],
        # '_bn_rm' -> '.running_mean'
        [r"(.*)bn.rm\Z", r"\1bn.running_mean"],
        # '_bn_riv' -> '.running_var'
        [r"(.*)bn.riv\Z", r"\1bn.running_var"],
        # '_b' -> '.bias'
        [r"(.*)[\._]b\Z", r"\1.bias"],
        # '_w' -> '.weight'
        [r"(.*)[\._]w\Z", r"\1.weight"],
    ]
    for source, dest in pairs:
        cf_key = re.sub(source, dest, cf_key)
    return cf_key


def convert_key_pt2ms(pt_key):
    # case1, add prefix 'model'
    pt_key = 'net.' + pt_key
    # case2, map batchnorm
    pt_key = pt_key.replace('bn.weight', 'bn.bn2d.gamma')
    pt_key = pt_key.replace('bn.bias', 'bn.bn2d.beta')
    pt_key = pt_key.replace('bn.running_mean', 'bn.bn2d.moving_mean')
    pt_key = pt_key.replace('bn.running_var', 'bn.bn2d.moving_variance')
    return pt_key


if __name__ == "__main__":
    convert_ckpt_cf2ms(cf_ckpt_path="SLOWFAST_8x8_R50.pkl")
