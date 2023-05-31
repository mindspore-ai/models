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

import os
from multiprocessing import Pool
import numpy as np
from PIL import Image

from common.option import TestOptions
from common.resize2d import SGKResize2dNumpy
from common.utils import cal_psnr, cal_ssim, _rgb2ycbcr

mode_pad_dict = {
    "s": 1,
    "d": 2,
    "y": 2,
    "c": 3,
    "t": 3,
}


# 4D equivalent of triangular interpolation, faster version
def simplex_interp_faster(
        weight, img_in, h, w, interval, rot, mode="s", o_c=1
):
    q = 2**interval
    l = 2 ** (8 - interval) + 1

    if mode == "s":
        # Extract MSBs
        img_a1 = img_in[:, 0 : 0 + h, 0 : 0 + w] // q
        img_b1 = img_in[:, 0 : 0 + h, 1 : 1 + w] // q
        img_c1 = img_in[:, 1 : 1 + h, 0 : 0 + w] // q
        img_d1 = img_in[:, 1 : 1 + h, 1 : 1 + w] // q

        # Extract LSBs
        fa = img_in[:, 0 : 0 + h, 0 : 0 + w] % q
        fb = img_in[:, 0 : 0 + h, 1 : 1 + w] % q
        fc = img_in[:, 1 : 1 + h, 0 : 0 + w] % q
        fd = img_in[:, 1 : 1 + h, 1 : 1 + w] % q

    elif mode == "d":
        img_a1 = img_in[:, 0 : 0 + h, 0 : 0 + w] // q
        img_b1 = img_in[:, 0 : 0 + h, 2 : 2 + w] // q
        img_c1 = img_in[:, 2 : 2 + h, 0 : 0 + w] // q
        img_d1 = img_in[:, 2 : 2 + h, 2 : 2 + w] // q
        fa = img_in[:, 0 : 0 + h, 0 : 0 + w] % q
        fb = img_in[:, 0 : 0 + h, 2 : 2 + w] % q
        fc = img_in[:, 2 : 2 + h, 0 : 0 + w] % q
        fd = img_in[:, 2 : 2 + h, 2 : 2 + w] % q

    elif mode == "y":
        img_a1 = img_in[:, 0 : 0 + h, 0 : 0 + w] // q
        img_b1 = img_in[:, 1 : 1 + h, 1 : 1 + w] // q
        img_c1 = img_in[:, 1 : 1 + h, 2 : 2 + w] // q
        img_d1 = img_in[:, 2 : 2 + h, 1 : 1 + w] // q
        fa = img_in[:, 0 : 0 + h, 0 : 0 + w] % q
        fb = img_in[:, 1 : 1 + h, 1 : 1 + w] % q
        fc = img_in[:, 1 : 1 + h, 2 : 2 + w] % q
        fd = img_in[:, 2 : 2 + h, 1 : 1 + w] % q

    elif mode == "c":
        img_a1 = img_in[:, 0 : 0 + h, 0 : 0 + w] // q
        img_b1 = img_in[:, 0 : 0 + h, 1 : 1 + w] // q
        img_c1 = img_in[:, 0 : 0 + h, 2 : 2 + w] // q
        img_d1 = img_in[:, 0 : 0 + h, 3 : 3 + w] // q
        fa = img_in[:, 0 : 0 + h, 0 : 0 + w] % q
        fb = img_in[:, 0 : 0 + h, 1 : 1 + w] % q
        fc = img_in[:, 0 : 0 + h, 2 : 2 + w] % q
        fd = img_in[:, 0 : 0 + h, 3 : 3 + w] % q

    elif mode == "t":
        img_a1 = img_in[:, 0 : 0 + h, 0 : 0 + w] // q
        img_b1 = img_in[:, 1 : 1 + h, 1 : 1 + w] // q
        img_c1 = img_in[:, 2 : 2 + h, 2 : 2 + w] // q
        img_d1 = img_in[:, 3 : 3 + h, 3 : 3 + w] // q
        fa = img_in[:, 0 : 0 + h, 0 : 0 + w] % q
        fb = img_in[:, 1 : 1 + h, 1 : 1 + w] % q
        fc = img_in[:, 2 : 2 + h, 2 : 2 + w] % q
        fd = img_in[:, 3 : 3 + h, 3 : 3 + w] % q
    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    p0000 = weight[
        img_a1.flatten().astype(np.int_) * l * l * l
        + img_b1.flatten().astype(np.int_) * l * l
        + img_c1.flatten().astype(np.int_) * l
        + img_d1.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p0001 = weight[
        img_a1.flatten().astype(np.int_) * l * l * l
        + img_b1.flatten().astype(np.int_) * l * l
        + img_c1.flatten().astype(np.int_) * l
        + img_d2.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p0010 = weight[
        img_a1.flatten().astype(np.int_) * l * l * l
        + img_b1.flatten().astype(np.int_) * l * l
        + img_c2.flatten().astype(np.int_) * l
        + img_d1.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p0011 = weight[
        img_a1.flatten().astype(np.int_) * l * l * l
        + img_b1.flatten().astype(np.int_) * l * l
        + img_c2.flatten().astype(np.int_) * l
        + img_d2.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p0100 = weight[
        img_a1.flatten().astype(np.int_) * l * l * l
        + img_b2.flatten().astype(np.int_) * l * l
        + img_c1.flatten().astype(np.int_) * l
        + img_d1.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p0101 = weight[
        img_a1.flatten().astype(np.int_) * l * l * l
        + img_b2.flatten().astype(np.int_) * l * l
        + img_c1.flatten().astype(np.int_) * l
        + img_d2.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p0110 = weight[
        img_a1.flatten().astype(np.int_) * l * l * l
        + img_b2.flatten().astype(np.int_) * l * l
        + img_c2.flatten().astype(np.int_) * l
        + img_d1.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p0111 = weight[
        img_a1.flatten().astype(np.int_) * l * l * l
        + img_b2.flatten().astype(np.int_) * l * l
        + img_c2.flatten().astype(np.int_) * l
        + img_d2.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))

    p1000 = weight[
        img_a2.flatten().astype(np.int_) * l * l * l
        + img_b1.flatten().astype(np.int_) * l * l
        + img_c1.flatten().astype(np.int_) * l
        + img_d1.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p1001 = weight[
        img_a2.flatten().astype(np.int_) * l * l * l
        + img_b1.flatten().astype(np.int_) * l * l
        + img_c1.flatten().astype(np.int_) * l
        + img_d2.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p1010 = weight[
        img_a2.flatten().astype(np.int_) * l * l * l
        + img_b1.flatten().astype(np.int_) * l * l
        + img_c2.flatten().astype(np.int_) * l
        + img_d1.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p1011 = weight[
        img_a2.flatten().astype(np.int_) * l * l * l
        + img_b1.flatten().astype(np.int_) * l * l
        + img_c2.flatten().astype(np.int_) * l
        + img_d2.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p1100 = weight[
        img_a2.flatten().astype(np.int_) * l * l * l
        + img_b2.flatten().astype(np.int_) * l * l
        + img_c1.flatten().astype(np.int_) * l
        + img_d1.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p1101 = weight[
        img_a2.flatten().astype(np.int_) * l * l * l
        + img_b2.flatten().astype(np.int_) * l * l
        + img_c1.flatten().astype(np.int_) * l
        + img_d2.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p1110 = weight[
        img_a2.flatten().astype(np.int_) * l * l * l
        + img_b2.flatten().astype(np.int_) * l * l
        + img_c2.flatten().astype(np.int_) * l
        + img_d1.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    p1111 = weight[
        img_a2.flatten().astype(np.int_) * l * l * l
        + img_b2.flatten().astype(np.int_) * l * l
        + img_c2.flatten().astype(np.int_) * l
        + img_d2.flatten().astype(np.int_)
    ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))

    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2]
    out = out.reshape(sz, -1)

    p0000 = p0000.reshape(sz, -1)
    p0100 = p0100.reshape(sz, -1)
    p1000 = p1000.reshape(sz, -1)
    p1100 = p1100.reshape(sz, -1)
    fa = fa.reshape(-1, 1)

    p0001 = p0001.reshape(sz, -1)
    p0101 = p0101.reshape(sz, -1)
    p1001 = p1001.reshape(sz, -1)
    p1101 = p1101.reshape(sz, -1)
    fb = fb.reshape(-1, 1)
    fc = fc.reshape(-1, 1)

    p0010 = p0010.reshape(sz, -1)
    p0110 = p0110.reshape(sz, -1)
    p1010 = p1010.reshape(sz, -1)
    p1110 = p1110.reshape(sz, -1)
    fd = fd.reshape(-1, 1)

    p0011 = p0011.reshape(sz, -1)
    p0111 = p0111.reshape(sz, -1)
    p1011 = p1011.reshape(sz, -1)
    p1111 = p1111.reshape(sz, -1)

    fab = fa > fb
    fac = fa > fc
    fad = fa > fd

    fbc = fb > fc
    fbd = fb > fd
    fcd = fc > fd

    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    out[i] = (
        (q - fa[i]) * p0000[i]
        + (fa[i] - fb[i]) * p1000[i]
        + (fb[i] - fc[i]) * p1100[i]
        + (fc[i] - fd[i]) * p1110[i]
        + (fd[i]) * p1111[i]
    )
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = (
        (q - fa[i]) * p0000[i]
        + (fa[i] - fb[i]) * p1000[i]
        + (fb[i] - fd[i]) * p1100[i]
        + (fd[i] - fc[i]) * p1101[i]
        + (fc[i]) * p1111[i]
    )
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(
        1
    )
    out[i] = (
        (q - fa[i]) * p0000[i]
        + (fa[i] - fd[i]) * p1000[i]
        + (fd[i] - fb[i]) * p1001[i]
        + (fb[i] - fc[i]) * p1101[i]
        + (fc[i]) * p1111[i]
    )
    i = np.logical_and.reduce(
        (~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)
    ).squeeze(1)

    out[i] = (
        (q - fd[i]) * p0000[i]
        + (fd[i] - fa[i]) * p0001[i]
        + (fa[i] - fb[i]) * p1001[i]
        + (fb[i] - fc[i]) * p1101[i]
        + (fc[i]) * p1111[i]
    )

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = (
        (q - fa[i]) * p0000[i]
        + (fa[i] - fc[i]) * p1000[i]
        + (fc[i] - fb[i]) * p1010[i]
        + (fb[i] - fd[i]) * p1110[i]
        + (fd[i]) * p1111[i]
    )
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = (
        (q - fa[i]) * p0000[i]
        + (fa[i] - fc[i]) * p1000[i]
        + (fc[i] - fd[i]) * p1010[i]
        + (fd[i] - fb[i]) * p1011[i]
        + (fb[i]) * p1111[i]
    )
    i7 = i = np.logical_and.reduce(
        (~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)
    ).squeeze(1)
    out[i] = (
        (q - fa[i]) * p0000[i]
        + (fa[i] - fd[i]) * p1000[i]
        + (fd[i] - fc[i]) * p1001[i]
        + (fc[i] - fb[i]) * p1011[i]
        + (fb[i]) * p1111[i]
    )
    i = np.logical_and.reduce(
        (~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)
    ).squeeze(1)
    out[i] = (
        (q - fd[i]) * p0000[i]
        + (fd[i] - fa[i]) * p0001[i]
        + (fa[i] - fc[i]) * p1001[i]
        + (fc[i] - fb[i]) * p1011[i]
        + (fb[i]) * p1111[i]
    )

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = (
        (q - fc[i]) * p0000[i]
        + (fc[i] - fa[i]) * p0010[i]
        + (fa[i] - fb[i]) * p1010[i]
        + (fb[i] - fd[i]) * p1110[i]
        + (fd[i]) * p1111[i]
    )
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(
        1
    )  # c > a > d > b
    out[i] = (
        (q - fc[i]) * p0000[i]
        + (fc[i] - fa[i]) * p0010[i]
        + (fa[i] - fd[i]) * p1010[i]
        + (fd[i] - fb[i]) * p1011[i]
        + (fb[i]) * p1111[i]
    )
    i11 = i = np.logical_and.reduce(
        (~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)
    ).squeeze(
        1
    )  # c > d > a > b
    out[i] = (
        (q - fc[i]) * p0000[i]
        + (fc[i] - fd[i]) * p0010[i]
        + (fd[i] - fa[i]) * p0011[i]
        + (fa[i] - fb[i]) * p1011[i]
        + (fb[i]) * p1111[i]
    )
    i = np.logical_and.reduce(
        (~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)
    ).squeeze(1)
    out[i] = (
        (q - fd[i]) * p0000[i]
        + (fd[i] - fc[i]) * p0001[i]
        + (fc[i] - fa[i]) * p0011[i]
        + (fa[i] - fb[i]) * p1011[i]
        + (fb[i]) * p1111[i]
    )

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = (
        (q - fb[i]) * p0000[i]
        + (fb[i] - fa[i]) * p0100[i]
        + (fa[i] - fc[i]) * p1100[i]
        + (fc[i] - fd[i]) * p1110[i]
        + (fd[i]) * p1111[i]
    )
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = (
        (q - fb[i]) * p0000[i]
        + (fb[i] - fa[i]) * p0100[i]
        + (fa[i] - fd[i]) * p1100[i]
        + (fd[i] - fc[i]) * p1101[i]
        + (fc[i]) * p1111[i]
    )
    i15 = i = np.logical_and.reduce(
        (~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)
    ).squeeze(1)
    out[i] = (
        (q - fb[i]) * p0000[i]
        + (fb[i] - fd[i]) * p0100[i]
        + (fd[i] - fa[i]) * p0101[i]
        + (fa[i] - fc[i]) * p1101[i]
        + (fc[i]) * p1111[i]
    )
    i = np.logical_and.reduce(
        (~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)
    ).squeeze(1)
    out[i] = (
        (q - fd[i]) * p0000[i]
        + (fd[i] - fb[i]) * p0001[i]
        + (fb[i] - fa[i]) * p0101[i]
        + (fa[i] - fc[i]) * p1101[i]
        + (fc[i]) * p1111[i]
    )

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = (
        (q - fb[i]) * p0000[i]
        + (fb[i] - fc[i]) * p0100[i]
        + (fc[i] - fa[i]) * p0110[i]
        + (fa[i] - fd[i]) * p1110[i]
        + (fd[i]) * p1111[i]
    )
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(
        1
    )
    out[i] = (
        (q - fb[i]) * p0000[i]
        + (fb[i] - fc[i]) * p0100[i]
        + (fc[i] - fd[i]) * p0110[i]
        + (fd[i] - fa[i]) * p0111[i]
        + (fa[i]) * p1111[i]
    )
    i19 = i = np.logical_and.reduce(
        (~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)
    ).squeeze(1)
    out[i] = (
        (q - fb[i]) * p0000[i]
        + (fb[i] - fd[i]) * p0100[i]
        + (fd[i] - fc[i]) * p0101[i]
        + (fc[i] - fa[i]) * p0111[i]
        + (fa[i]) * p1111[i]
    )
    i = np.logical_and.reduce(
        (~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)
    ).squeeze(1)
    out[i] = (
        (q - fd[i]) * p0000[i]
        + (fd[i] - fb[i]) * p0001[i]
        + (fb[i] - fc[i]) * p0101[i]
        + (fc[i] - fa[i]) * p0111[i]
        + (fa[i]) * p1111[i]
    )

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = (
        (q - fc[i]) * p0000[i]
        + (fc[i] - fb[i]) * p0010[i]
        + (fb[i] - fa[i]) * p0110[i]
        + (fa[i] - fd[i]) * p1110[i]
        + (fd[i]) * p1111[i]
    )
    i22 = i = np.logical_and.reduce(
        (~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)
    ).squeeze(1)
    out[i] = (
        (q - fc[i]) * p0000[i]
        + (fc[i] - fb[i]) * p0010[i]
        + (fb[i] - fd[i]) * p0110[i]
        + (fd[i] - fa[i]) * p0111[i]
        + (fa[i]) * p1111[i]
    )
    i23 = i = np.logical_and.reduce(
        (~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)
    ).squeeze(1)
    out[i] = (
        (q - fc[i]) * p0000[i]
        + (fc[i] - fd[i]) * p0010[i]
        + (fd[i] - fb[i]) * p0011[i]
        + (fb[i] - fa[i]) * p0111[i]
        + (fa[i]) * p1111[i]
    )
    i = np.logical_and.reduce(
        (~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])
    ).squeeze(1)
    out[i] = (
        (q - fd[i]) * p0000[i]
        + (fd[i] - fc[i]) * p0001[i]
        + (fc[i] - fb[i]) * p0011[i]
        + (fb[i] - fa[i]) * p0111[i]
        + (fa[i]) * p1111[i]
    )

    out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], o_c))
    out = np.transpose(out, (0, 3, 1, 2)).reshape(
        (img_a1.shape[0] * o_c, img_a1.shape[1], img_a1.shape[2])
    )
    out = np.rot90(out, rot, [1, 2])
    out = out / q
    return out


class Eltr:
    def __init__(self, arg, lut_dict):
        self.arg = arg
        self.lut_dict = lut_dict
        self.norm = 255
        self.out_c = 3  # hyper channel
        self.modes = arg.modes
        self.modes2 = arg.modes2
        self.stages = arg.stages
        self.files = None
        self.dataset = None
        self.scale_h = 0
        self.scale_w = 0
        self.result_path = None
        self.resizer = SGKResize2dNumpy(support_sz=arg.suppSize, max_sigma=arg.sigma)

    def run(self, dataset, scale_h, scale_w, num_worker=24):
        folder = os.path.join(self.arg.testDir, dataset, "HR")
        files = os.listdir(folder)
        files.sort()

        result_path = os.path.join(
            opt.resultRoot,
            opt.expDir.split("/")[-1],
            "X{:.2f}_{:.2f}".format(scale_h, scale_w),
            dataset,
        )
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        self.result_path = result_path
        self.dataset = dataset
        self.files = files
        self.scale_h = scale_h
        self.scale_w = scale_w
        pool = Pool(num_worker)
        psnr_ssim_list = pool.map(self._worker, list(range(len(self.files))))
        pool.close()
        pool.join()
        return psnr_ssim_list

    def _worker(self, i):
        img_lr = np.array(
            Image.open(
                os.path.join(self.arg.testDir, self.dataset,
                             "LR_bicubic/X{}".format(self.scale_h), self.files[i]))
        ).astype(np.float32)
        if len(img_lr.shape) == 2:
            img_lr = np.expand_dims(img_lr, axis=2)
            img_lr = np.concatenate([img_lr, img_lr, img_lr], axis=2)
        img_gt = np.array(
            Image.open(os.path.join(self.arg.testDir, self.dataset, "HR", self.files[i])))
        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate([img_gt, img_gt, img_gt], axis=2)

        for s in range(self.stages - 1):
            pred = 0
            stage = s + 1
            for mode_i in self.modes:
                key = "s{}_{}r0".format(str(stage), mode_i)
                weight = self.lut_dict[key]
                pad = mode_pad_dict.get(mode_i, "")
                for r in [0, 1, 2, 3]:
                    img_lr_rot = np.rot90(img_lr, r)
                    h, w, _ = img_lr_rot.shape
                    img_in = np.pad(
                        img_lr_rot, ((0, pad), (0, pad), (0, 0)), mode="edge"
                    ).transpose((2, 0, 1))
                    pred += simplex_interp_faster(weight, img_in, h, w,
                                                  self.arg.interval, 4 - r,
                                                  mode=mode_i, o_c=1)
            if (s + 1) == (self.stages - 1):
                avg_factor, bias, norm = len(self.modes), 0, 1
            else:
                avg_factor, bias, norm = (len(self.modes) * 4, self.norm // 2,
                                          float(self.norm))
            img_lr = (
                np.round(np.clip((pred / avg_factor) + bias, 0, self.norm))
                .astype(np.float32)
                .transpose((1, 2, 0)))  # feat [C, H, W]
        pred = 0
        for mode_j in self.modes2:
            pad = mode_pad_dict.get(mode_j, "")
            for r in [0, 2]:
                key = "s{}_{}r0".format(str(self.stages), mode_j)
                weight = self.lut_dict[key]
                img_lr_rot = np.rot90(img_lr, r)
                h, w, _ = img_lr_rot.shape
                img_in = np.pad(
                    img_lr_rot, ((0, pad), (0, pad), (0, 0)), mode="edge"
                ).transpose((2, 0, 1))
                pred += simplex_interp_faster(weight, img_in, h, w,
                                              self.arg.interval, 4 - r,
                                              mode=mode_j, o_c=self.out_c)
            for r in [1, 3]:
                key = "s{}_{}r1".format(str(self.stages), mode_j)
                weight = self.lut_dict[key]
                img_lr_rot = np.rot90(img_lr, r)
                h, w, _ = img_lr_rot.shape
                img_in = np.pad(
                    img_lr_rot, ((0, pad), (0, pad), (0, 0)), mode="edge"
                ).transpose((2, 0, 1))
                pred += simplex_interp_faster(weight, img_in, h, w,
                                              self.arg.interval, 4 - r,
                                              mode=mode_j, o_c=self.out_c)

        avg_factor, bias, norm = len(self.modes2) * 4, self.norm // 2, float(self.norm)
        img_hyper = (
            np.round(np.clip((pred / avg_factor) + bias, 0, self.norm)).astype(
                np.float32) / norm)  # [C, H, W]
        img_lr = img_lr.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
        self.resizer.set_shape(img_lr.shape, scale_factors=[self.scale_h, self.scale_w])
        c, _, _ = img_hyper.shape
        hyper1_c_idx = list(range(0, c, 3))
        hyper2_c_idx = list(range(1, c + 1, 3))
        hyper3_c_idx = list(range(2, c + 2, 3))
        img_out = self.resizer.resize(img_lr, img_hyper[hyper1_c_idx, :, :],
                                      img_hyper[hyper2_c_idx, :, :], img_hyper[hyper3_c_idx, :, :],)
        img_out = np.clip(np.round(img_out).transpose((1, 2, 0)), 0, self.norm).astype(
            np.uint8)
        Image.fromarray(img_out).save(
                        os.path.join(self.result_path,
                                     "{}_{}_{}bit.png".format(self.files[i].split("/")[-1][:-4],
                                                              self.arg.lutName, 8 - self.arg.interval,),))
        img_lr = np.clip(np.round(img_lr).transpose((1, 2, 0)), 0, self.norm).astype(
            np.uint8)
        Image.fromarray(img_lr).save(
                        os.path.join(self.result_path,
                                     "{}_{}_{}lr.png".format(self.files[i].split("/")[-1][:-4],
                                                             self.arg.lutName, 8 - self.arg.interval)))
        if img_gt.shape != img_out.shape:
            pred_h, pred_w, _ = img_out.shape
            img_gt = img_gt[:pred_h, :pred_w, :]
        y_gt, y_out = _rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(img_out)[:, :, 0]
        psnr = cal_psnr(y_gt, y_out, max(int(self.scale_h), int(self.scale_w)))
        ssim = cal_ssim(y_gt, y_out)
        return [psnr, ssim]


if __name__ == "__main__":
    opt = TestOptions().parse()

    # Load LUT
    lut_dict_load = dict()
    for s_i in range(opt.stages):
        stage_i = s_i + 1
        cur_modes = opt.modes
        rots = ["0"]
        out_c = 1
        if stage_i == opt.stages:  # hyper stage
            cur_modes = opt.modes2
            rots = ["0", "1"]
            out_c = 3
        for mode_cur in cur_modes:
            for r_i in rots:
                key_i = "s{}_{}r{}".format(str(s_i + 1), mode_cur, r_i)
                lutPath = os.path.join(
                    opt.expDir,
                    "{}_{}bit_int8_s{}_{}r{}.npy".format(
                        opt.lutName, 8 - opt.interval, str(stage_i), mode_cur, r_i
                    ),
                )
                lut_dict_load[key_i] = (
                    np.array(np.load(lutPath)).astype(np.float32).reshape(-1, out_c)
                )

    etr = Eltr(opt, lut_dict_load)

    all_datasets = ["Set5"]
    all_scales = [
        [2, 2],
        [3, 3],
        [4, 4],
    ]

    scale_head = ["Scale".ljust(15, " ")]
    for scale_p in all_scales:
        scale_h_j, scale_w_j = scale_p
        scale_head.append("{:.1f}x{:.1f}\t".format(scale_h_j, scale_w_j))
    print("\t".join(scale_head))

    for dataset_i in all_datasets:
        metric_list = [dataset_i.ljust(15, " ")]
        for scale_p in all_scales:
            scale_h_i, scale_w_i = scale_p
            psnr_ssim_s = etr.run(dataset_i, scale_h_i, scale_w_i)
            avg_psnr, avg_ssim = np.mean(np.asarray(psnr_ssim_s)[:, 0]), np.mean(
                np.asarray(psnr_ssim_s)[:, 1]
            )
            metric_list.append("{:.2f}/{:.4f}".format(avg_psnr, avg_ssim))
        print("\t".join(metric_list))
