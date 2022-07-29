
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
"""
Add notes.
"""


import math
import h5py
import numpy as np
import cv2
from src.config import config
import mindspore
from mindspore import ops, Tensor, nn
from scipy import interpolate


def load_mean_theta():
    mean = np.zeros(config.total_theta_count, dtype=np.float)

    mean_values = h5py.File(config.smpl_mean_theta_path)

    mean_pose = mean_values['pose'][:]
    mean_pose[:3] = 0
    mean_shape = mean_values['shape'][:]
    mean_pose[0] = np.pi

    mean[0] = 0.9

    mean[3:75] = mean_pose[:]
    mean[75:] = mean_shape[:]
    return mean


def batch_rodrigues(theta):

    net = nn.Norm(axis=1)
    l1norm = net(theta + 1e-8)
    expand_dims = ops.ExpandDims()
    angle = expand_dims(l1norm, -1)
    div = ops.Div()
    normalized = div(theta, angle)
    angle = angle * 0.5
    cos = ops.Cos()
    v_cos = cos(angle)
    sin = ops.Sin()
    v_sin = sin(angle)
    op = ops.Concat(1)
    quat = op([v_cos, v_sin * normalized])

    return quat2mat(quat)


def quat2mat(quat):

    norm_quat = quat
    net = nn.Norm(axis=1, keep_dims=True)
    norm_quat = norm_quat / net(norm_quat)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    B = quat.shape[0]
    pow_ = ops.Pow()
    w2, x2, y2, z2 = pow_(w, 2), pow_(x, 2), pow_(y, 2), pow_(z, 2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    stack = ops.Stack(1)
    rotMat = stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ]).view(B, 3, 3)
    return rotMat


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    N = Rs.shape[0]
    N = Tensor(N, mindspore.int32)
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                            dtype=np.float32)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = np_rot_x
        root_rotation = ops.matmul(Tensor(Rs[:, 0, :, :]), Tensor(rot_x))
    else:
        root_rotation = Rs[:, 0, :, :]
    expand_dims = ops.ExpandDims()
    Js = expand_dims(Js, -1)

    def make_A(R, t):
        R_homo = Tensor(np.zeros((R.shape[0], R.shape[1] + 1, R.shape[2])),
                        config.type)
        pad_op = nn.Pad(paddings=((0, 1), (0, 0)))
        for i in range(R.shape[0]):
            R_homo[i] = pad_op(R[i]).copy()

        ones = ops.Ones()
        op = ops.Concat(1)
        t = Tensor(t, config.type)
        t_homo = op((t, ones((N, 1, 1), config.type)))
        op = ops.Concat(2)
        return op((R_homo, t_homo))

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = ops.matmul(Tensor(results[parent[i]]), Tensor(A_here))

        results.append(res_here)

    stack = ops.Stack(1)

    results = stack(results)
    new_J = results[:, :, :3, 3]
    op = ops.Concat(2)
    zeros = ops.Zeros()

    Js_w0 = op((Js, zeros((N, 24, 1, 1), config.type)))
    init_bone = ops.matmul(results, Js_w0)
    pad_op = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (3, 0)))
    init_bone = pad_op(init_bone)

    A = results - init_bone
    return new_J, A


def cal_temp_ab(a, b):
    b = b.view(-1, 1, 3)
    a_trans = a[:, :, :2] + b[:, :, 1:]
    shape = a_trans.shape
    return (b[:, :, 0] * a_trans.view(shape[0], -1)).view(shape)


def calc_temp_ab2(a):
    if not a:
        return False, False, False

    temp_a = np.array([a[0][0], a[0][1]])
    temp_ = temp_a.copy()
    for pt in a:
        temp_a[0] = min(temp_a[0], pt[0])
        temp_a[1] = min(temp_a[1], pt[1])
        temp_[0] = max(temp_[0], pt[0])
        temp_[1] = max(temp_[1], pt[1])

    return temp_a, temp_, len(a) >= 5


def calc_obb(a):
    ca = np.cov(a, y=None, rowvar=0, bias=1)
    _, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)
    ar = np.dot(a, np.linalg.inv(tvect))
    a_min = np.min(ar, axis=0)
    a_max = np.max(ar, axis=0)
    d_f = (a_max - a_min) * 0.5
    cen_temp = a_min + d_f
    corn_temp = np.array([
        cen_temp + [-d_f[0], -d_f[1]], cen_temp + [d_f[0], -d_f[1]],
        cen_temp + [d_f[0], d_f[1]], cen_temp + [-d_f[0], d_f[1]]
    ])
    corn_temp = np.dot(corn_temp, tvect)
    return corn_temp


def getcutb(l_t, r_b, ExpRatio, Cter_=None):

    def _expcrbox(l_t, r_b, scale):
        Cter_ = (l_t + r_b) / 2.0
        a, b, c, d = l_t[0] - Cter_[0], r_b[0] - Cter_[0], l_t[1] - Cter_[
            1], r_b[1] - Cter_[1]
        a, b, c, d = a * scale[0], b * scale[1], c * scale[
            2], d * scale[3]
        l_t, r_b = np.array([Cter_[0] + a, Cter_[1] + c
                             ]), np.array([Cter_[0] + b, Cter_[1] + d])
        lb, rt = np.array([Cter_[0] + a, Cter_[1] + d
                           ]), np.array([Cter_[0] + b, Cter_[1] + c])
        Cter_ = (l_t + r_b) / 2
        tmp_ = [Cter_, l_t, rt, r_b, lb]
        return tmp_

    if Cter_ is None:
        Cter_ = (l_t + r_b) // 2
    tmp = _expcrbox(
        l_t, r_b, ExpRatio)

    Cter_, l_t, _, r_b, _ = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]

    tmp_oft = (r_b - l_t) // 2

    tmp_xc = tmp_oft[0]
    tmp_xy = tmp_oft[1]

    r = max(tmp_xc, tmp_xy)

    tmp_xc = r
    tmp_xy = r

    x = int(Cter_[0])
    y = int(Cter_[1])

    return [x - tmp_xc, y - tmp_xy], [x + tmp_xc, y + tmp_xy]


def off_set_pts(k_p, l_t):

    res_ = k_p.copy()
    res_[:, 0] -= l_t[0]
    res_[:, 1] -= l_t[1]
    return res_


def cut_image(input_img, kps, exp_r, l_top, r_tom):

    input_img = input_img
    img_h = input_img.shape[0]
    img_w = input_img.shape[1]
    img_chann = input_img.shape[2] if len(input_img.shape) >= 3 else 1

    l_top, r_tom = getcutb(l_top, r_tom,
                           exp_r)

    l_t = [int(l_top[0]), int(l_top[1])]
    r_b = [int(r_tom[0]), int(r_tom[1])]

    l_t[0] = max(0, l_t[0])
    l_t[1] = max(0, l_t[1])
    r_b[0] = min(r_b[0], img_w)
    r_b[1] = min(r_b[1], img_h)

    l_top = [int(l_top[0]), int(l_top[1])]
    r_tom = [int(r_tom[0] + 0.5), int(r_tom[1] + 0.5)]

    dstImage = np.zeros(shape=[
        r_tom[1] - l_top[1], r_tom[0] - l_top[0], img_chann],
                        dtype=np.uint8)
    dstImage[:, :, :] = 0

    tmp_ = [l_t[0] - l_top[0], l_t[1] - l_top[1]]
    size = [r_b[0] - l_t[0], r_b[1] - l_t[1]]

    dstImage[tmp_[1]:size[1] + tmp_[1],
             tmp_[0]:size[0] + tmp_[0], :] = input_img[l_t[1]:r_b[1],
                                                       l_t[0]:r_b[0], :]
    return dstImage, off_set_pts(kps, l_top)


def reflect_lsp_kp(tmp_keypoints):
    tmp_kp_ = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
    joref = tmp_keypoints[tmp_kp_]
    joref[:, 0] = -joref[:, 0]

    return joref - np.mean(joref, axis=0)


def reflect_pose(a):
    index = np.array([
        0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18, 19,
        20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37,
        38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58,
        59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68
    ])

    siflip = np.array([
        1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
        -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
        -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
        1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1
    ])
    return a[index] * siflip


def crop_image(image_path, angle, l_t, r_b, tmp_scale, kp_2d, tmp_crze):

    def _expcr_box(l_t, r_b, tmp_scale):
        tmp_cen = (l_t + r_b) / 2.0
        a, b, c, d = l_t[0] - tmp_cen[0], r_b[0] - \
            tmp_cen[0], l_t[1] - tmp_cen[1], r_b[1] - tmp_cen[1]
        a, b, c, d = a * tmp_scale[0], b * tmp_scale[1], c * tmp_scale[
            2], d * tmp_scale[3]
        l_t, r_b = np.array([tmp_cen[0] + a, tmp_cen[1] + c
                             ]), np.array([tmp_cen[0] + b, tmp_cen[1] + d])
        tmp_lb, tmp_rt = np.array([tmp_cen[0] + a, tmp_cen[1] + d
                                   ]), np.array([tmp_cen[0] + b, tmp_cen[1] + c])
        tmp_cen = (l_t + r_b) / 2
        tmp_ = [tmp_cen, l_t, tmp_rt, r_b, tmp_lb]
        return tmp_

    def _extbox(tmp_cen, l_t, tmp_rt, r_b, tmp_lb, tmp_crze):
        tmp_lx, tmp_ly = np.linalg.norm(
            tmp_rt - l_t), np.linalg.norm(tmp_lb - l_t)
        tmp_dx, tmp_dy = (tmp_rt - l_t) / tmp_lx, (tmp_lb - l_t) / tmp_ly
        l = max(tmp_lx, tmp_ly) / 2.0
        tmp__ = [
            tmp_cen - l * tmp_dx - l * tmp_dy,
            tmp_cen + l * tmp_dx - l * tmp_dy,
            tmp_cen + l * tmp_dx + l * tmp_dy,
            tmp_cen - l * tmp_dx + l * tmp_dy,
            tmp_dx,
            tmp_dy,
            tmp_crze * 1.0 / l]
        return tmp__

    def _gsampoints(l_t, tmp_rt, r_b, tmp_lb, tmp_crze):
        vectmp_x = tmp_rt - l_t
        vectmp_y = tmp_lb - l_t
        index_x, index_y = np.meshgrid(range(tmp_crze), range(tmp_crze))
        index_x = index_x.astype(np.float)
        index_y = index_y.astype(np.float)
        index_x /= float(tmp_crze)
        index_y /= float(tmp_crze)
        tmp_intepots = index_x[..., np.newaxis].repeat(
            2, axis=2) * vectmp_x + index_y[..., np.newaxis].repeat(2, axis=2) * vectmp_y
        tmp_intepots += l_t
        return tmp_intepots

    def _samage(tmp_simg, tmp_intepots):
        tmp_inimg = np.zeros(
            (tmp_intepots.shape[0] * tmp_intepots.shape[1],
             tmp_simg.shape[2]))
        index_x = range(tmp_simg.shape[1])
        index_y = range(tmp_simg.shape[0])
        tmp_flainpoints = tmp_intepots.reshape(
            [tmp_intepots.shape[0] * tmp_intepots.shape[1], 2])
        for tmp_ic in range(tmp_simg.shape[2]):
            tmp_inimg[:, tmp_ic] = interpolate.interpn(
                (index_y, index_x),
                tmp_simg[:, :, tmp_ic],
                tmp_flainpoints[:, [1, 0]],
                method='nearest',
                bounds_error=False,
                fill_value=0)
        tmp_inimg = tmp_inimg.reshape(
            (tmp_intepots.shape[0], tmp_intepots.shape[1],
             tmp_simg.shape[2]))

        return tmp_inimg

    def _trkp2d(kps, tmp_cen, tmp_dx, tmp_dy, l_t, tmp_rat):
        tmp_k2off = kps[:, :2] - tmp_cen
        tmp_prox, tmp_proy = np.dot(
            tmp_k2off, tmp_dx), np.dot(
                tmp_k2off, tmp_dy)
        for tmp_iidx in range(len(kps)):
            kps[tmp_iidx, :2] = (
                tmp_dx * tmp_prox[tmp_iidx] + tmp_dy * tmp_proy[tmp_iidx] + l_t) * tmp_rat
        return kps

    tmp_simg = cv2.imread(image_path)
    tmp = _expcr_box(l_t, r_b, tmp_scale)
    tmp_cen, l_t, tmp_rt, r_b, tmp_lb = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]

    radian = angle * np.pi / 180.0
    v_sin, v_cos = math.sin(radian), math.cos(radian)

    rot_matrix = np.array([[v_cos, v_sin], [-v_sin, v_cos]])

    n_corner = (np.dot(rot_matrix,
                       np.array([l_t - tmp_cen,
                                 tmp_rt - tmp_cen,
                                 r_b - tmp_cen,
                                 tmp_lb - tmp_cen]).T).T) + tmp_cen
    n_l_t, n_tmp_rt, n_r_b, n_tmp_lb = n_corner[0], n_corner[1], n_corner[2], n_corner[3]

    corners = calc_obb(
        np.array([l_t, tmp_rt, r_b, tmp_lb, n_l_t, n_tmp_rt, n_r_b, n_tmp_lb]))
    l_t, tmp_rt, r_b, tmp_lb = corners[0], corners[1], corners[2], corners[3]
    tmp = _extbox(tmp_cen,
                  l_t,
                  tmp_rt,
                  r_b,
                  tmp_lb,
                  tmp_crze=tmp_crze)
    l_t, tmp_rt, r_b, tmp_lb, tmp_dx, tmp_dy, tmp_rat = tmp[
        0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6]
    s_pts = _gsampoints(l_t, tmp_rt, r_b, tmp_lb, tmp_crze)
    dst_image = _samage(tmp_simg, s_pts)
    kp_2d = _trkp2d(kp_2d, tmp_cen, tmp_dx, tmp_dy, l_t, tmp_rat)

    return dst_image, kp_2d


def flip_image(src_image, kps):
    _, w = src_image.shape[0], src_image.shape[1]
    src_image = cv2.flip(src_image, 1)
    if kps is not None:
        kps[:, 0] = w - 1 - kps[:, 0]
        kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]

    return src_image, kps[kp_map]


def lint_tm(sa, sb):
    al, ar, bl, br = sa[0], sa[1], sb[0], sb[1]
    assert al <= ar and bl <= br
    if al >= br or bl >= ar:
        return False
    return True


def rectangle_intersect(in_a, in_b):
    a1 = [in_a[0][0], in_a[1][0]]
    a2 = [in_a[0][1], in_a[1][1]]

    b1 = [in_b[0][0], in_b[1][0]]
    b2 = [in_b[0][1], in_b[1][1]]

    return lint_tm(a1, b1) and lint_tm(a2, b2)


def g_inrectangle(a, b, c, d):
    if not rectangle_intersect([a, b], [c, d]):
        return None, None

    lt = a.copy()
    rb = b.copy()

    lt[0] = max(lt[0], c[0])
    lt[1] = max(lt[1], c[1])

    rb[0] = min(rb[0], d[0])
    rb[1] = min(rb[1], d[1])
    return lt, rb


def g_utangle(a, b, c, d):
    lt = a.copy()
    rb = b.copy()

    lt[0] = min(lt[0], c[0])
    lt[1] = min(lt[1], c[1])

    rb[0] = max(rb[0], d[0])
    rb[1] = max(rb[1], d[1])
    return lt, rb


def get_rectangle_area(a, b):
    return (b[0] - a[0]) * (b[1] - a[1])


def get_rectangle_intersect_ratio(a, b, c, d):
    (a, b), (c, d) = g_inrectangle(
        a, b, c, d), g_utangle(a, b, c, d)

    if a is None:
        _ = 0.0
    else:
        _ = 1.0 * get_rectangle_area(a, b) / get_rectangle_area(
            c, d)
    return _


def align_by_pelvis(a):
    id_1 = 3
    id_2 = 2
    reshape = ops.Reshape()
    a = reshape(a, (a.shape[0], 14, 3))
    b = (a[:, id_1, :] + a[:, id_2, :]) / 2.0
    expand_dims = ops.ExpandDims()
    return a - expand_dims(b, 1)
