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
"""layer utils"""
import sys
from pathlib import Path
from typing import List, Tuple
import mindspore as ms
from mindspore import nn, ops, Tensor

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../pointnet2_lib/src")

tt = open("log.txt", "a")


def log_to_file(s: str):
    """log to file"""
    tt.write(str(s))
    tt.write('\n')
    tt.flush()


op_map = {}


def get_func_from_so(so_name: str,
                     func_name: str,
                     output_n=-1,
                     CPU_opt=False,
                     out_shape=None,
                     out_dtype=None,
                     in_type=None):
    """get function from so"""
    if not func_name.startswith("ms_"):
        func_name = "ms_" + func_name
    k = hash((func_name, out_shape, out_dtype, CPU_opt, output_n, in_type))

    sos: List[Path] = list(Path(__file__).parent.parent.glob("**/*.so"))
    for so in sos:
        if so_name in so.name:
            name = f"{so.absolute()}:{func_name}"
            t = lambda *x: x[output_n]
            if out_shape and out_dtype:
                op = ops.Custom(name,
                                out_shape=out_shape,
                                out_dtype=out_dtype,
                                func_type="aot")
            else:
                op = ops.Custom(name,
                                out_shape=t,
                                out_dtype=t,
                                func_type="aot")
            if CPU_opt is True:
                op.add_prim_attr("primitive_target", "CPU")
            op_map[k] = op
            return op
    raise Exception(f"Can't find {so_name} in {sos}")


class _ConvBase(nn.Cell):
    """Conv Base"""

    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size,
                 stride,
                 padding,
                 activation,
                 bn,
                 conv=None,
                 batch_norm=None,
                 bias=True,
                 preact=False,
                 name="",
                 instance_norm=False,
                 instance_norm_func=None):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(in_size,
                         out_size,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         has_bias=bias)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False)
        if preact:
            if bn:
                self.insert_child_to_cell(name + 'bn', bn_unit)

            if activation is not None:
                self.insert_child_to_cell(name + 'activation', activation)

            if not bn and instance_norm:
                self.insert_child_to_cell(name + 'in', in_unit)
        self.insert_child_to_cell(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.insert_child_to_cell(name + 'bn', bn_unit)

            if activation is not None:
                self.insert_child_to_cell(name + 'activation', activation)

            if not bn and instance_norm:
                self.insert_child_to_cell(name + 'in', in_unit)


class Conv1d(_ConvBase):
    """Conv1d"""

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 activation=nn.ReLU(),
                 bn: bool = False,
                 bias: bool = True,
                 preact: bool = False,
                 name: str = "",
                 instance_norm=False):
        super().__init__(in_size,
                         out_size,
                         kernel_size,
                         stride,
                         padding,
                         activation,
                         bn,
                         conv=nn.Conv1d,
                         batch_norm=nn.BatchNorm2d,
                         bias=bias,
                         preact=preact,
                         name=name,
                         instance_norm=instance_norm,
                         instance_norm_func=nn.InstanceNorm1d)

    def construct(self, inputs):
        x = inputs
        for cell in self.cells():
            if not isinstance(cell, nn.BatchNorm2d):
                x = cell(x)
            else:
                x = x.expand_dims(-1)
                x = cell(x)
                x = x.squeeze(-1)
        return x


class Conv2d(_ConvBase):
    """Conv2d"""

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 activation=nn.ReLU(),
                 bn: bool = False,
                 bias: bool = True,
                 preact: bool = False,
                 name: str = "",
                 instance_norm=False):
        super().__init__(in_size,
                         out_size,
                         kernel_size,
                         stride,
                         padding,
                         activation,
                         bn,
                         conv=nn.Conv2d,
                         batch_norm=nn.BatchNorm2d,
                         bias=bias,
                         preact=preact,
                         name=name,
                         instance_norm=instance_norm,
                         instance_norm_func=nn.InstanceNorm2d)

    def construct(self, inputs):
        x = inputs
        for cell in self.cells():
            x = cell(x)
        return x


class SharedMLP(nn.Cell):
    """SahreMLP"""

    def __init__(self,
                 args: List[int],
                 *,
                 bn: bool = False,
                 activation=nn.ReLU(),
                 preact: bool = False,
                 first: bool = False,
                 instance_norm: bool = False):
        super(SharedMLP, self).__init__()
        cells = []
        for i in range(len(args) - 1):
            cells.append(
                Conv2d(args[i],
                       args[i + 1],
                       bn=(not first or not preact or (i != 0)) and bn,
                       activation=activation if
                       (not first or not preact or (i != 0)) else None,
                       preact=preact,
                       instance_norm=instance_norm))
        self.celllist = nn.SequentialCell(cells)

    def construct(self, inputs):
        return self.celllist(inputs)


class GatherOperation(nn.Cell):
    """GatherOperation"""

    def __init__(self):
        super().__init__()
        self.func_name = 'gather_points_wrapper_fast'
        self.back_func_name = "gather_points_grad_wrapper_fast"
        self.so_name = "pointnet2_cuda"

    def construct(self, features: ms.Tensor, idx: ms.Tensor) -> ms.Tensor:
        """
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """

        B, npoint = idx.shape
        _, C, N = features.shape

        log_to_file("GatherOperation")
        log_to_file((B, C, npoint))
        op = get_func_from_so(so_name=self.so_name,
                              func_name=self.func_name,
                              out_shape=(B, C, npoint),
                              out_dtype=ms.float32)
        _B = Tensor(B, ms.dtype.int32)
        _C = Tensor(C, ms.dtype.int32)
        _N = Tensor(N, ms.dtype.int32)
        _npoint = Tensor(npoint, ms.dtype.int32)
        # print(_B, _C, _N, _npoint)
        output = op(_B, _C, _N, _npoint, features, idx)
        return output


class GroupingOperation(nn.Cell):
    """GroupingOperation"""

    def __init__(self):
        super(GroupingOperation, self).__init__()
        self.so_name = "pointnet2_cuda"
        self.func_name = "group_points_wrapper_fast"
        self.back_func_name = "group_points_grad_wrapper"

    def construct(self, features: ms.Tensor, idx: ms.Tensor) -> ms.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indices of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """

        _, C, N = features.shape
        B, npoint, nsample = idx.shape

        log_to_file("GroupingOperation")
        log_to_file((B, C, npoint, nsample))
        op = get_func_from_so(so_name=self.so_name,
                              func_name=self.func_name,
                              out_shape=(B, C, npoint, nsample),
                              out_dtype=ms.float32)
        _B = ms.Tensor(B, ms.int32)
        _C = ms.Tensor(C, ms.int32)
        _N = ms.Tensor(N, ms.int32)
        _npoint = ms.Tensor(npoint, ms.int32)
        _nsample = ms.Tensor(nsample, ms.int32)
        # print("GroupingOperation")
        # print(_B, _C, _N, _npoint, _nsample)
        output = op(_B, _C, _N, _npoint, _nsample, features, idx)
        return output


grouping_operation = GroupingOperation()


class BallQuery(nn.Cell):
    """BallQuery"""


    def construct(self, radius: float, nsample: int, xyz: ms.Tensor,
                  new_xyz: ms.Tensor) -> ms.Tensor:
        """
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indices of the features that form the query balls
        """

        B, N, _ = xyz.shape
        npoint = new_xyz.shape[1]

        ball_query_wrapper = get_func_from_so("pointnet2_cuda",
                                              "ball_query_wrapper_fast",
                                              out_shape=(B, npoint, nsample),
                                              out_dtype=ms.int32)

        idx = ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz)
        return idx


ball_query = BallQuery()


class QueryAndGroup(nn.Cell):
    """QueryAndGroup"""

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def construct(self,
                  xyz: ms.Tensor,
                  new_xyz: ms.Tensor,
                  features: ms.Tensor = None) -> Tuple[ms.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        assert new_xyz.shape[2] == 3
        ball_query_ = BallQuery()
        idx = ball_query_(self.radius, self.nsample, xyz, new_xyz)

        xyz_trans = xyz.swapaxes(1, 2)
        grouped_xyz = grouping_operation(xyz_trans,
                                         idx)  # (B, 3, npoint, nsample)

        grouped_xyz -= new_xyz.swapaxes(1, 2).expand_dims(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)

            if self.use_xyz:
                new_features = ops.Concat(1)([grouped_xyz, grouped_features
                                              ])  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz
        return new_features


class FurthestPointSampling(nn.Cell):
    """FurthestPointSampling"""


    def construct(self, xyz: ms.Tensor, npoint: int) -> ms.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """

        B, N, _ = xyz.shape

        temp = ms.numpy.full((B, N), 1e10)
        log_to_file("FurthestPointSampling")
        log_to_file((B, npoint))
        furthest_point_sampling_wrapper = get_func_from_so(
            "pointnet2_cuda",
            "furthest_point_sampling_wrapper",
            out_shape=(B, npoint),
            out_dtype=ms.int32)
        _B = ms.Tensor(B, ms.int32)
        _N = ms.Tensor(N, ms.int32)
        _npoint = ms.Tensor(npoint, ms.int32)
        output = furthest_point_sampling_wrapper(_B, _N, _npoint, xyz, temp)
        return output


furthest_point_sample = FurthestPointSampling()


class _PointnetSAModuleBase(nn.Cell):
    """_PointnetSAModuleBase"""

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def construct(self,
                  xyz: ms.Tensor,
                  features: ms.Tensor = None,
                  new_xyz=None) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        if features is not None:
            # print('features: ', features.mean())
            pass

        xyz_flipped = xyz.swapaxes(1, 2)
        if new_xyz is None:
            new_xyz = GatherOperation()(
                xyz_flipped, furthest_point_sample(xyz, self.npoint)).swapaxes(
                    1, 2) if self.npoint is not None else None

        for i in range(len(self.groupers)):

            new_features = self.groupers[i](
                xyz, new_xyz, features)  # (B, C, npoint, nsample)
            new_features = self.mlps[i](
                new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = nn.MaxPool2d(
                    kernel_size=(1, new_features.shape[3]))(
                        new_features)  # (B, mlp[-1], npoint, 1) 待检验
            elif self.pool_method == 'avg_pool':
                new_features = nn.AvgPool2d(
                    kernel_size=(1, new_features.size(3)))(
                        new_features)  # (B, mlp[-1], npoint, 1) 待检验
            else:
                raise NotImplementedError

            new_features = ops.Squeeze(-1)(new_features)
            new_features_list.append(new_features)

        return new_xyz, ops.Concat(1)(new_features_list)


class GroupAll(nn.Cell):
    """GroupAll"""

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def construct(self,
                  xyz: ms.Tensor,
                  new_xyz: ms.Tensor,
                  features: ms.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        # print(new_xyz)
        grouped_xyz = xyz.swapaxes(1, 2).expand_dims(2)
        if features is not None:

            grouped_features = ops.expand_dims(features, 2)
            if self.use_xyz:
                new_features = ops.Concat(1)([grouped_xyz, grouped_features
                                              ])  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self,
                 *,
                 npoint: int,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 instance_norm=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.CellList()
        self.mlps = nn.CellList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz
                              ) if npoint is not None else GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(
                SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self,
                 *,
                 mlp: List[int],
                 npoint: int = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 instance_norm=False):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__(mlps=[mlp],
                         npoint=npoint,
                         radii=[radius],
                         nsamples=[nsample],
                         bn=bn,
                         use_xyz=use_xyz,
                         pool_method=pool_method,
                         instance_norm=instance_norm)


class ThreeNN(nn.Cell):
    """ThreeNN"""

    def construct(self, unknown: Tensor,
                  known: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """

        B, N, _ = unknown.shape
        m = known.shape[1]
        three_nn_wrapper = get_func_from_so(
            "pointnet2_cuda.cpython-39-x86_64-linux-gnu.so",
            "three_nn_wrapper_fast",
            out_shape=((B, N, 3), (B, N, 3)),
            out_dtype=(ms.float32, ms.int32))

        _B = ms.Tensor(B, ms.int32)
        _N = ms.Tensor(N, ms.int32)
        _m = ms.Tensor(m, ms.int32)
        dist2, idx = three_nn_wrapper(_B, _N, _m, unknown, known)
        return ops.sqrt(dist2), idx


class ThreeInterpolate(nn.Cell):
    """ThreeInterpolate"""

    def __init__(self):
        super(ThreeInterpolate, self).__init__()
        self.so_name = "pointnet2_cuda.cpython-39-x86_64-linux-gnu.so"

    def construct(self, features: Tensor, idx: Tensor,
                  weight: Tensor) -> Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """

        B, c, m = features.shape
        n = idx.shape[1]
        log_to_file("ThreeInterpolate")
        log_to_file((B, c, n))
        three_interpolate_wrapper = get_func_from_so(
            self.so_name,
            "three_interpolate_wrapper_fast",
            out_shape=(B, c, n),
            out_dtype=ms.float32)
        _B = ms.Tensor(B, ms.int32)
        _c = ms.Tensor(c, ms.int32)
        _m = ms.Tensor(m, ms.int32)
        _n = ms.Tensor(n, ms.int32)
        output = three_interpolate_wrapper(_B, _c, _m, _n, features, idx,
                                           weight)
        return output


class PointnetFPModule(nn.Cell):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = SharedMLP(mlp, bn=bn)
        self.three_nn = ThreeNN()
        self.three_interpolate = ThreeInterpolate()

    def construct(self, unknown: Tensor, known: Tensor, unknow_feats: Tensor,
                  known_feats: Tensor) -> Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = self.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)

            norm = ops.ReduceSum(keep_dims=True)(dist_recip, axis=2)
            weight = dist_recip / norm

            interpolated_feats = self.three_interpolate(
                known_feats, idx, weight)
        else:
            interpolated_feats = ops.broadcast_to(
                known_feats, (*known_feats.shape[0:2], unknown.shape[1]))

        if unknow_feats is not None:
            new_features = ops.concat([interpolated_feats, unknow_feats],
                                      axis=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats
        new_features = new_features.expand_dims(-1)
        new_features: Tensor = self.mlp(new_features)

        return new_features.squeeze(-1)
