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

from mindspore import ops


def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = ops.norm(theta + 1e-8, dim=1)
    angle = ops.ExpandDims()(l1norm, -1)
    normalized = ops.Div()(theta, angle)
    angle = angle*0.5
    v_cos = ops.Cos()(angle)
    v_sin = ops.Sin()(angle)
    quat = ops.Concat(axis=1)((v_cos, v_sin*normalized))
    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    y = ops.norm(norm_quat, dim=1, keepdim=True)
    x = norm_quat

    norm_quat = x / y
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.shape[0]

    w2, x2, y2, z2 = ops.Pow()(w, 2), ops.Pow()(x, 2), ops.Pow()(y, 2), ops.Pow()(z, 2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = ops.Stack(axis=1)([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                                2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                                2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2]).view(B, 3, 3)
    return rotMat


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2]+camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d
