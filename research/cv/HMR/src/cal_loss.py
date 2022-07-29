
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
from src.config import config
from mindspore.common import dtype as mstype
from mindspore import nn, Tensor, ops


class CalcLossD(nn.Cell):
    def __init__(self, discriminator):
        super(CalcLossD, self).__init__()
        self.discriminator = discriminator
        self.bacth_size_2d = config.batch_size
        self.bacth_size_3d = config.batch_3d_size
        self.bacth_size_mosh = config.adv_batch_size
        self.e_loss_weight = Tensor(config.e_loss_weight, config.type)
        self.e_3d_loss_weight = Tensor(config.e_3d_loss_weight, config.type)
        self.e_3d_kp_ratio = Tensor(config.e_3d_kp_ratio, config.type)
        self.d_loss_weight = Tensor(config.d_loss_weight, config.type)
        self.e_shape_ratio = Tensor(config.e_shape_ratio, config.type)
        self.e_pose_ratio = Tensor(config.e_pose_ratio, config.type)
        self.d_disc_ratio = Tensor(config.d_disc_ratio, config.type)
        self.square = ops.Square()
        self.oprm = ops.ReduceSum(keep_dims=True)

    def construct(
            self,
            discriminator_outputs,
            discriminator_outputs_None=None):

        disc_f = discriminator_outputs[0:
                                       self.bacth_size_2d + self.bacth_size_3d]
        disc_r = discriminator_outputs[self.bacth_size_2d +
                                       self.bacth_size_3d:]
        disc_1, disc_2, disc_3 = self.cal_disc_loss(
            disc_r, disc_f)
        disc_1, disc_2, disc_3 = disc_1 * self.d_loss_weight * self.d_disc_ratio, disc_2 * \
            self.d_loss_weight * self.d_disc_ratio, disc_3 * self.d_loss_weight * self.d_disc_ratio
        disc_3 = disc_3 / self.d_disc_ratio
        return disc_3

    def cal_disc_loss(self, disc_r, disc_f):
        tmp_a = disc_r.shape[0]
        tmp_b = disc_f.shape[0]
        b_l, l_a = self.oprm(disc_f**2) / tmp_b, self.oprm(
            (disc_r - 1)**2) / tmp_a
        return l_a, b_l, l_a + b_l


class CalcLossG(nn.LossBase):
    def __init__(self, discriminator):

        super(CalcLossG, self).__init__()
        self.discriminator = discriminator
        self.oprm = ops.ReduceSum(keep_dims=True)
        self.op_0 = ops.Concat(0)
        self.abs = ops.Abs()
        self.e_loss_weight = Tensor(config.e_loss_weight, config.type)
        self.e_3d_loss_weight = Tensor(config.e_3d_loss_weight, config.type)
        self.e_3d_kp_ratio = Tensor(config.e_3d_kp_ratio, config.type)
        self.d_loss_weight = Tensor(config.d_loss_weight, config.type)
        self.e_shape_ratio = Tensor(config.e_shape_ratio, config.type)
        self.e_pose_ratio = Tensor(config.e_pose_ratio, config.type)
        self.d_disc_ratio = Tensor(config.d_disc_ratio, config.type)
        self.bacth_size_2d = config.batch_size
        self.bacth_size_3d = config.batch_3d_size
        self.bacth_size_mosh = config.adv_batch_size
        self.pow = ops.Pow()
        self.expand_dims = ops.ExpandDims()
        self.ones = ops.Ones()
        self.stack_1 = ops.Stack(1)
        self.stack_2 = ops.Stack(2)
        self.op_2 = ops.Concat(2)
        self.op_1 = ops.Concat(1)
        self.div = ops.Div()
        self.cos = ops.Cos()
        self.sin = ops.Sin()
        self.net_1 = nn.Norm(axis=1)
        self.net_2 = nn.Norm(axis=1, keep_dims=True)
        self.sub = ops.Sub()
        self.zeros = ops.Zeros()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()

    def construct(self, generator_outputs, data_2d_label, data_3d_label):

        data_2d_label = data_2d_label.reshape(self.bacth_size_2d, -1, 3)

        data_3d_theta, w_3d, w_smpl = data_3d_label[:, 84:84 + 85].copy(
        ), data_3d_label[:, -1].copy(), data_3d_label[:, -2].copy()

        total_predict_thetas = generator_outputs[0]

        predict_theta, predict_j2d, predict_j3d = generator_outputs[
            0], generator_outputs[2], generator_outputs[3]

        real_2d, real_3d = self.op_0((data_2d_label, data_3d_label[:, :42].reshape(
            self.bacth_size_3d, -1, 3))), data_3d_label[:, 42:42 + 42].reshape(self.bacth_size_3d, -1, 3)

        predict_j2d, predict_j3d, predict_theta = predict_j2d, predict_j3d[
            self.bacth_size_2d:, :], predict_theta[self.bacth_size_2d:, :]

        loss_kp_2d = self.kp_l1_loss(
            real_2d, predict_j2d[:, :14, :]) * self.e_loss_weight

        loss_kp_3d = self.cal_3d_loss(
            real_3d, predict_j3d[:, :14, :], w_3d) * self.e_3d_loss_weight * self.e_3d_kp_ratio

        real_shape, predict_shape = data_3d_theta[:,
                                                  75:], predict_theta[:, 75:]

        loss_shape = self.batch_shape_l2_loss(
            real_shape,
            predict_shape,
            w_smpl) * self.e_3d_loss_weight * self.e_shape_ratio

        real_pose, predict_pose = data_3d_theta[:,
                                                3:75], predict_theta[:, 3:75]

        loss_pose = self.batch_pose_l2_loss(
            real_pose, predict_pose, w_smpl) * self.e_3d_loss_weight * self.e_pose_ratio

        e_disc_loss = self.encoder_l2_loss(self.discriminator(
            total_predict_thetas)) * self.d_loss_weight * self.d_disc_ratio

        e_loss = loss_kp_2d + loss_kp_3d + loss_shape + loss_pose + e_disc_loss
        return e_loss

    def encoder_l2_loss(self, disc_value):

        return self.oprm((disc_value - 1.0)**2) * 1.0 / \
            (self.bacth_size_2d + self.bacth_size_3d)

    def kp_l1_loss(self, real_kp, predict_2d_kp):

        kp_1 = real_kp.view(-1, 3)
        kp_2 = predict_2d_kp.copy().view(-1, 2)
        vis_ = kp_1[:, 2]
        n = self.oprm(vis_) * 2.0 + 1e-8
        abs_12 = self.abs(kp_1[:, :2] - kp_2).sum(1)
        return ops.matmul(abs_12, vis_) * 1.0 / n

    def cal_3d_loss(self, real_3d_kp, fake_3d_kp, w_3d):

        k = self.oprm(w_3d) * 42 * 3.0 * 2.0 + 1e-8
        real_3d_kp, fake_3d_kp = self.align_by_pelvis(
            real_3d_kp), self.align_by_pelvis(fake_3d_kp)
        kp_dif = (real_3d_kp - fake_3d_kp)**2
        return ops.matmul(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k

    def batch_shape_l2_loss(self, real_shape, fake_shape, w_shape):

        k = self.oprm(w_shape) * 10.0 * 2.0 + 1e-8
        shape_dif = (real_shape - fake_shape)**2
        return ops.matmul(shape_dif.sum(1), w_shape) * 1.0 / k

    def batch_pose_l2_loss(self, real_pose, fake_pose, w_pose):
        k = self.oprm(w_pose) * 207.0 * 2.0 + 1e-8
        real_rs, fake_rs = self.batch_rodrigues(real_pose.view(-1, 3)).view(
            -1, 24, 9)[:, 1:, :], self.batch_rodrigues(fake_pose.view(-1,
                                                                      3)).view(-1, 24,
                                                                               9)[:, 1:, :]
        dif_rs = ((real_rs - fake_rs)**2).view(-1, 207)
        return self.cast(
            ops.matmul(
                self.cast(
                    dif_rs.sum(1),
                    mstype.float16),
                self.cast(
                    w_pose,
                    mstype.float16)),
            mstype.float32) * 1.0 / k

    def quat2mat(self, quat):

        norm_quat = quat
        norm_quat = norm_quat / self.net_2(norm_quat)
        w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                                1], norm_quat[:,
                                                              2], norm_quat[:,
                                                                            3]

        w2, x2, y2, z2 = self.pow(
            w, 2), self.pow(
                x, 2), self.pow(
                    y, 2), self.pow(
                        z, 2)

        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        rotMat = self.stack_1([
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
            2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
            2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
        ]).view(self.bacth_size_2d * 24, 3, 3)
        return rotMat

    def batch_rodrigues(self, theta):

        l1norm = self.net_1(theta + 1e-8)
        angle = self.expand_dims(l1norm, -1)
        Normalizationd = self.div(theta, angle)
        angle = angle * 0.5
        v_cos = self.cos(angle)
        v_sin = self.sin(angle)
        quat = self.op_1([v_cos, v_sin * Normalizationd])

        return self.quat2mat(quat)

    def align_by_pelvis(self, joints):

        joints = self.reshape(joints, (self.bacth_size_2d, 14, 3))
        pelvis = (joints[:, 3, :] + joints[:, 2, :]) / 2.0
        return joints - self.expand_dims(pelvis, 1)
