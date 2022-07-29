
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
import sys
import pickle
from src.config import config
from src import util
import numpy as np
import mindspore.numpy as mnp
from mindspore import ops, Tensor, nn, Parameter
from mindspore.ops import operations as P


class SMPL(nn.Cell):
    def __init__(self, model_path, joint_type='cocoplus', obj_saveable=False):
        super(SMPL, self).__init__()
        self.type = config.type
        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(
                joint_type)
            sys.exit(msg)
        self.model_path = model_path
        self.joint_type = joint_type

        with open(model_path, 'rb') as reader:
            model = pickle.load(reader, encoding='iso-8859-1')

        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None

        np_v_template = np.array(model['v_template'], dtype=np.float32)
        self.v_template = Parameter(Tensor((np_v_template), self.type),
                                    requires_grad=False,
                                    name="smpl_v_template")
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float32)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.shapedirs = Parameter(Tensor(np_shapedirs, self.type),
                                   requires_grad=False,
                                   name="smpl_shapedirs")

        np_J_regressor = np.array(model['J_regressor'].toarray(),
                                  dtype=np.float32)
        self.J_regressor = Parameter(Tensor(np_J_regressor, self.type),
                                     requires_grad=False,
                                     name="smpl_J_regressor")

        np_posedirs = np.array(model['posedirs'], dtype=np.float32)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.posedirs = Parameter(Tensor(np_posedirs, self.type),
                                  requires_grad=False,
                                  name="smpl_posedirs")

        self.parents = [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
            18, 19, 20, 21
        ]

        np_joint_regressor = Tensor(
            np.array(model['cocoplus_regressor'].toarray(), dtype=np.float32),
            self.type)
        if joint_type == 'lsp':
            self.joint_regressor = Parameter(Tensor(np_joint_regressor[:, :14],
                                                    self.type),
                                             requires_grad=False,
                                             name="smpl_joint_regressor_14")
        else:
            self.joint_regressor = Parameter(Tensor(np_joint_regressor,
                                                    self.type),
                                             requires_grad=False,
                                             name="smpl_joint_regressor")

        np_weights = np.array(model['weights'], dtype=np.float32)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        batch_size = max(config.batch_size + config.batch_3d_size,
                         config.eval_batch_size)
        np_weights = np.tile(np_weights, (batch_size, 1))
        self.weight = Tensor(
            np_weights.reshape(-1, vertex_count, vertex_component), self.type)
        self.eye = ops.Eye()
        self.e3 = self.eye(3, 3, self.type)

        self.cur_device = None
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
        self.batch_size = config.batch_size + config.batch_3d_size
        self.np_rot_x = Tensor(
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
            self.type)
        self.R_homo = Parameter(Tensor(np.zeros((self.batch_size, 4, 3)),
                                       self.type),
                                requires_grad=False,
                                name="smpl_jR_homo")

    def batch_global_rigid_transformation(self, Rs, Js, rotate_base=False):
        if rotate_base:
            np_rot_x = mnp.reshape(
                mnp.tile(self.np_rot_x, [self.batch_size, 1]),
                [self.batch_size, 3, 3])
            root_rotation = ops.matmul(Rs[:, 0, :, :], np_rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]
        Js = self.expand_dims(Js, -1)

        def make_A(R, t):

            pad_op = nn.Pad(paddings=((0, 1), (0, 0)))
            for i in range(self.batch_size):
                self.R_homo[i] = pad_op(R[i]).copy()

            t_homo = self.op_1((t, self.ones((self.batch_size, 1, 1),
                                             self.type)))
            return self.op_2((self.R_homo, t_homo))

        #
        A0 = make_A(root_rotation, Js[:, 0])

        results = [A0]

        for i in range(1, 24):
            j_here = Js[:, i] - Js[:, self.parents[i]]
            A_here = make_A(Rs[:, i], j_here)

            res_here = ops.matmul(results[self.parents[i]], A_here)

            results.append(res_here)

        results = self.stack_1(results)
        new_J = results[:, :, :3, 3]
        Js_w0 = self.op_2(
            (Js, self.zeros((self.batch_size, 24, 1, 1), self.type)))
        init_bone = ops.matmul(results, Js_w0)
        pad_op = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (3, 0)))
        init_bone = pad_op(init_bone)
        A = results - init_bone
        return new_J, A

    def quat2mat(self, quat):
        """Convert quaternion coefficients to rotation matrix.
        config:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / self.net_2(norm_quat)
        w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                                1], norm_quat[:,
                                                              2], norm_quat[:,
                                                                            3]

        w2, x2, y2, z2 = self.pow(w,
                                  2), self.pow(x,
                                               2), self.pow(y,
                                                            2), self.pow(z, 2)

        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        rotMat = self.stack_1([
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
            2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
            2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
        ]).view(self.batch_size * 24, 3, 3)
        return rotMat

    def batch_rodrigues(self, theta):

        l1norm = self.net_1(theta + 1e-8)
        angle = self.expand_dims(l1norm, -1)
        normalized = self.div(theta, angle)
        angle = angle * 0.5
        v_cos = self.cos(angle)
        v_sin = self.sin(angle)
        quat = self.op_1([v_cos, v_sin * normalized])

        return self.quat2mat(quat)

    def construct(self, beta, theta, get_skin=True):

        v_shaped = ops.matmul(beta,
                              self.shapedirs).view(-1,
                                                   self.size[0],
                                                   self.size[1]) + self.v_template
        Jx = ops.matmul(v_shaped[:, :, 0], self.J_regressor.T)
        Jy = ops.matmul(v_shaped[:, :, 1], self.J_regressor.T)
        Jz = ops.matmul(v_shaped[:, :, 2], self.J_regressor.T)
        J = self.stack_2([Jx, Jy, Jz])
        Rs = self.batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = self.sub(Rs[:, 1:, :, :], self.e3).view(-1, 207)
        v_posed = ops.matmul(pose_feature, self.posedirs).view(
            -1, self.size[0], self.size[1]) + v_shaped
        _, A = self.batch_global_rigid_transformation(
            Rs, J, rotate_base=True)

        T = ops.matmul(self.weight[:self.batch_size],
                       A.view(self.batch_size, 24,
                              16)).view(self.batch_size, -1, 4, 4)

        v_posed_homo = self.op_2(
            (v_posed,
             self.ones((self.batch_size, v_posed.shape[1], 1), self.type)))
        v_homo = ops.matmul(T, self.expand_dims(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joint_x = ops.matmul(verts[:, :, 0], self.joint_regressor.T)
        joint_y = ops.matmul(verts[:, :, 1], self.joint_regressor.T)
        joint_z = ops.matmul(verts[:, :, 2], self.joint_regressor.T)
        joints = self.stack_2((joint_x, joint_y, joint_z))

        if get_skin:
            tmp = [verts, joints, Rs]
        else:
            tmp = joints
        return tmp


def conv3x3(in_channels, out_channels, stride=1, padding=1, pad_mode='pad'):
    """3x3 convolution """
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     pad_mode=pad_mode)


def conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='pad'):
    """1x1 convolution"""
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     stride=stride,
                     padding=padding,
                     pad_mode=pad_mode)


class ResidualBlock(nn.Cell):
    """
    residual Block
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = down_sample

        self.conv_down_sample = conv1x1(in_channels,
                                        out_channels,
                                        stride=stride,
                                        padding=0)
        self.bn_down_sample = nn.BatchNorm2d(out_channels)
        self.add = P.Add()

    def construct(self, x):
        """
        :param x:
        :return:
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class PRNetEncoder(nn.Cell):
    """
    resnet nn.Cell
    """

    def __init__(self, block):
        super(PRNetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

        self.layer1 = self.MakeLayer(block,
                                     3,
                                     in_channels=64,
                                     out_channels=256,
                                     stride=1)
        self.layer2 = self.MakeLayer(block,
                                     4,
                                     in_channels=256,
                                     out_channels=512,
                                     stride=2)
        self.layer3 = self.MakeLayer(block,
                                     6,
                                     in_channels=512,
                                     out_channels=1024,
                                     stride=2)
        self.layer4 = self.MakeLayer(block,
                                     3,
                                     in_channels=1024,
                                     out_channels=2048,
                                     stride=2)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.flatten = P.Flatten()

    def MakeLayer(self, block, layer_num, in_channels, out_channels, stride):
        """
        make block layer
        :param block:
        :param layer_num:
        :param in_channels:
        :param out_channels:
        :param stride:
        :return:
        """
        layers = []
        resblk = block(in_channels,
                       out_channels,
                       stride=stride,
                       down_sample=True)
        layers.append(resblk)

        for _ in range(1, layer_num):
            resblk = block(out_channels, out_channels, stride=1)
            layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x


def load_Res50Model():
    model = PRNetEncoder(ResidualBlock)
    return model


class ThetaRegressor(nn.Cell):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func,
                 iterations):
        super(ThetaRegressor, self).__init__()
        self.line = LinearModel(fc_layers, use_dropout, drop_prob, use_ac_func)
        self.iterations = iterations
        self.batch_size = config.batch_size + config.batch_3d_size
        mean_theta = np.tile(
            util.load_mean_theta(), self.batch_size).reshape(
                (self.batch_size, -1))
        self.mean_theta = Tensor(mean_theta, config.type)
        self.op1 = ops.Concat(1)

    def construct(self, inputs):
        thetas = []
        theta = self.mean_theta[:self.batch_size, :]

        for _ in range(self.iterations):

            total_inputs = self.op1((inputs, theta))
            theta_temp = self.line(total_inputs)
            theta = theta + theta_temp
            thetas.append(theta)
        return thetas[-1]


class HMRNetBase(nn.Cell):
    def __init__(self):
        super(HMRNetBase, self).__init__()
        self._read_configs()
        print('start creating sub modules...')
        self._create_sub_modules()

    def _read_configs(self):

        self.batch_size = config.batch_size + config.batch_3d_size
        self.encoder_name = config.encoder_network
        self.beta_count = config.beta_count
        self.smpl_model = config.smpl_model
        self.smpl_mean_theta_path = config.smpl_mean_theta_path
        self.total_theta_count = config.total_theta_count
        self.joint_count = config.joint_count
        self.feature_count = config.feature_count

    def _create_sub_modules(self):
        '''
            ddd smpl model, SMPL can create a mesh from beta & theta
        '''
        self.smpl = SMPL(self.smpl_model, obj_saveable=True)

        self.encoder = load_Res50Model()

        fc_layers = [
            self.feature_count + self.total_theta_count, 1024, 1024, 85
        ]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]  # unactive the last layer

        self.iterations = 3
        self.regressor = ThetaRegressor(fc_layers, use_dropout, drop_prob,
                                        use_ac_func, self.iterations)

        print('finished create the encoder modules...')

    def cal_temp_ab(self, X, camera):
        '''
            X is N x num_points x 3
        '''
        camera = camera.view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        return (camera[:, :, 0] *
                X_trans.view(self.batch_size, -
                             1)).view(X_trans.shape)

    def construct(self, inputs):

        feature = self.encoder(inputs)

        theta = self.regressor(feature)

        cam = theta[:, 0:3].copy()

        pose = theta[:, 3:75].copy()
        shape = theta[:, 75:].copy()

        verts, j3d, Rs = self.smpl(beta=shape, theta=pose)

        j2d = self.cal_temp_ab(j3d, cam)

        out = [theta, verts, j2d, j3d, Rs]
        return out


class HMRNetBaseExport(nn.Cell):
    def __init__(self):
        super(HMRNetBaseExport, self).__init__()
        self._read_configs()
        print('start creating sub modules...')
        self._create_sub_modules()

    def _read_configs(self):

        self.batch_size = config.batch_size + config.batch_3d_size
        self.encoder_name = config.encoder_network
        self.beta_count = config.beta_count
        self.smpl_model = config.smpl_model
        self.smpl_mean_theta_path = config.smpl_mean_theta_path
        self.total_theta_count = config.total_theta_count
        self.joint_count = config.joint_count
        self.feature_count = config.feature_count

    def _create_sub_modules(self):
        '''
            ddd smpl model, SMPL can create a mesh from beta & theta
        '''
        self.smpl = SMPL(self.smpl_model, obj_saveable=True)

        self.encoder = load_Res50Model()

        fc_layers = [
            self.feature_count + self.total_theta_count, 1024, 1024, 85
        ]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]  # unactive the last layer

        self.iterations = 3
        self.regressor = ThetaRegressor(fc_layers, use_dropout, drop_prob,
                                        use_ac_func, self.iterations)

        print('finished create the encoder modules...')

    def cal_temp_ab(self, X, camera):
        '''
            X is N x num_points x 3
        '''
        camera = camera.view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        return (camera[:, :, 0] *
                X_trans.view(self.batch_size, -
                             1)).view(X_trans.shape)

    def construct(self, inputs):

        feature = self.encoder(inputs)

        theta = self.regressor(feature)

        cam = theta[:, 0:3].copy()

        pose = theta[:, 3:75].copy()
        shape = theta[:, 75:].copy()

        _, j3d, _ = self.smpl(beta=shape, theta=pose)

        _ = self.cal_temp_ab(j3d, cam)

        return j3d


class LinearModel(nn.Cell):

    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        super(LinearModel, self).__init__()
        self.fc_layers = fc_layers
        self.use_dropout = use_dropout
        self.drop_prob = drop_prob
        self.use_ac_func = use_ac_func
        self.line = self.create_layers()

    def create_layers(self):
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_use_ac_func = len(self.use_ac_func)

        layers = []
        for _ in range(l_fc_layer - 1):

            layers.append(nn.Dense(self.fc_layers[_], self.fc_layers[_ + 1]))
            if _ < l_use_ac_func and self.use_ac_func[_]:

                layers.append(nn.ReLU())

            if _ < l_use_drop and self.use_dropout[_]:

                layers.append(nn.Dropout(keep_prob=self.drop_prob[_]))

        return nn.SequentialCell(layers)

    def construct(self, x):

        return self.line(x)


class ShapeDiscriminator(nn.Cell):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        super(ShapeDiscriminator, self).__init__()
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(
                fc_layers[-1])
            sys.exit(msg)
        self.line = LinearModel(fc_layers, use_dropout, drop_prob, use_ac_func)

    def construct(self, inputs):
        return self.line(inputs)


class PoseDiscriminator(nn.Cell):
    def __init__(self, channels):
        super(PoseDiscriminator, self).__init__()

        if channels[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(
                channels[-1])
            sys.exit(msg)

        back_ = []
        l = len(channels)
        for idx in range(l - 2):
            back_.append(
                nn.Conv2d(in_channels=channels[idx],
                          out_channels=channels[idx + 1],
                          kernel_size=1,
                          stride=1,
                          has_bias=True))

        self.conv_blocks = nn.SequentialCell(back_)
        lin_layers = []
        for idx in range(23):

            lin_layers.append(
                nn.Dense(in_channels=channels[l - 2], out_channels=1))
        self.fc_layer = nn.CellList(lin_layers)

    def construct(self, inputs):

        transpose = ops.Transpose()
        expand_dims = ops.ExpandDims()
        inputs = expand_dims(transpose(inputs, (0, 2, 1)),
                             2)  # to N x 9 x 1 x 23
        internal_outputs = self.conv_blocks(inputs)
        o = []
        for idx in range(23):
            o.append(self.fc_layer[idx](internal_outputs[:, :, 0, idx]))
        op = ops.Concat(1)
        return op(o), internal_outputs


class FullPoseDiscriminator(nn.Cell):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        super(FullPoseDiscriminator, self).__init__()
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(
                fc_layers[-1])
            sys.exit(msg)
        self.line = LinearModel(fc_layers, use_dropout, drop_prob, use_ac_func)

    def construct(self, inputs):
        return self.line(inputs)


class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()
        self._read_configs()
        self._create_sub_modules()
        self.pow = ops.Pow()

    def _read_configs(self):
        self.beta_count = config.beta_count
        self.smpl_model = config.smpl_model
        self.smpl_mean_theta_path = config.smpl_mean_theta_path
        self.total_theta_count = config.total_theta_count
        self.joint_count = config.joint_count
        self.feature_count = config.feature_count

    def _create_sub_modules(self):
        '''
            create theta discriminator for 23 joint
        '''
        self.pose_discriminator = PoseDiscriminator([9, 32, 32, 1])
        fc_layers = [23 * 32, 1024, 1024, 1]
        use_dropout = [False, False, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        self.full_pose_discriminator = FullPoseDiscriminator(
            fc_layers, use_dropout, drop_prob, use_ac_func)
        fc_layers = [self.beta_count, 5, 1]
        use_dropout = [False, False]
        drop_prob = [0.5, 0.5]
        use_ac_func = [True, False]
        self.shape_discriminator = ShapeDiscriminator(fc_layers, use_dropout,
                                                      drop_prob, use_ac_func)

        print('finished create the discriminator modules...')

    def quat2mat(self, quat):

        norm_quat = quat
        net = nn.Norm(axis=1, keep_dims=True)
        norm_quat = norm_quat / net(norm_quat)
        w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                                1], norm_quat[:,
                                                              2], norm_quat[:,
                                                                            3]

        B = quat.shape[0]

        w2, x2, y2, z2 = self.pow(
            w, 2), self.pow(
                x, 2), self.pow(
                    y, 2), self.pow(z, 2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        stack = ops.Stack(1)
        rotMat = stack([
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
            2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
            2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
        ]).view(B, 3, 3)
        return rotMat

    def batch_rodrigues(self, theta):
        net = nn.Norm(axis=1)
        l1norm = net(theta + 1e-8)
        expand_dims = ops.ExpandDims()
        angle = expand_dims(l1norm, -1)
        div = ops.Div()
        Normalizationd = div(theta, angle)
        angle = angle * 0.5
        cos = ops.Cos()
        v_cos = cos(angle)
        sin = ops.Sin()
        v_sin = sin(angle)
        op = ops.Concat(1)
        quat = op([v_cos, v_sin * Normalizationd])

        return self.quat2mat(quat)

    def construct(self, thetas):
        '''
        inputs is N x 85(3 + 72 + 10)
        '''
        batch_size = thetas.shape[0]
        poses, shapes = thetas[:, 3:75], thetas[:, 75:]
        shape_disc_value = self.shape_discriminator(shapes)
        rotate_matrixs = self.batch_rodrigues(
            poses.copy().view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
        pose_disc_value, pose_inter_disc_value = self.pose_discriminator(
            rotate_matrixs)
        full_pose_disc_value = self.full_pose_discriminator(
            pose_inter_disc_value.copy().view(batch_size, -1))
        op = ops.Concat(1)
        return op((pose_disc_value, full_pose_disc_value, shape_disc_value))
