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

import mindspore.numpy as np
from mindspore import ops


def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: shape = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- shape = [B, 3, 3]
    """
    l1norm = np.norm(theta + 1e-8, ord=2, axis=1)
    angle = np.expand_dims(l1norm, -1)
    normalized = np.divide(theta, angle)
    angle = angle * 0.5
    v_cos = np.cos(angle)
    v_sin = np.sin(angle)
    quat = np.concatenate((v_cos, v_sin * normalized), axis=1)
    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: shape = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- shape = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / np.norm(norm_quat, ord=2, axis=1, keepdims=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.shape[0]
    pow_ = ops.Pow()

    w2, x2, y2, z2 = pow_(w, 2), pow_(x, 2), pow_(y, 2), pow_(z, 2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    stack_op = ops.Stack(axis=1)
    rotMat = stack_op([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2]).view(B, 3, 3)
    return rotMat


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: shape = [B, N, 3]
        camera: shape = [B, 3]
    Returns:
        Projected 2D points -- shape = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d
