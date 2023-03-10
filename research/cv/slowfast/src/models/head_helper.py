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

"""ResNe(X)t Head helper."""
import numpy as np
from mindspore import Tensor, ops, nn
from mindspore import numpy as mnp


class ResNetRoIHead(nn.Cell):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
            self,
            dim_in,
            num_classes,
            pool_size,
            resolution,
            scale_factor,
            dropout_rate=0.0,
            act_func="softmax",
            aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.pool_size = pool_size
        self.resolution = resolution
        self.scale_factor = scale_factor
        for pathway in range(self.num_pathways):
            spatial_pool = nn.MaxPool2d(tuple(resolution[pathway]), stride=1)
            self.insert_child_to_cell(
                "s{}_spool".format(pathway), spatial_pool)

        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=1 - dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        in_features = sum(dim_in)
        out_features = num_classes
        k = 1 / in_features
        fc_weight = Tensor(np.random.uniform(-k**0.5, k**0.5, (out_features, in_features)).astype(np.float32))
        fc_bias = Tensor(np.random.uniform(-k**0.5, k**0.5, (out_features)).astype(dtype=np.float32))
        self.projection = nn.Dense(
            sum(dim_in), num_classes, has_bias=True, weight_init=fc_weight, bias_init=fc_bias)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def construct(self, inputs, bboxes):
        """Construction."""
        n, pad_size = bboxes.shape[:2]
        index = mnp.arange(n).astype(bboxes.dtype).reshape((n, 1, 1))
        index = mnp.tile(index, (1, pad_size, 1))
        bboxes = ops.Concat(axis=2)((index, bboxes)).reshape(n * pad_size, -1)

        pool_out = []
        s0_tpool = ops.ReduceMean(keep_dims=False)
        s1_tpool = ops.ReduceMean(keep_dims=False)
        s0_roi = ops.ROIAlign(
            self.resolution[0][0],
            self.resolution[0][1],
            spatial_scale=1.0 / self.scale_factor[0],
            sample_num=2,
            roi_end_mode=1
        )
        s1_roi = ops.ROIAlign(
            self.resolution[1][0],
            self.resolution[1][1],
            spatial_scale=1.0 / self.scale_factor[1],
            sample_num=2,
            roi_end_mode=1
        )

        out = s0_tpool(inputs[0], 2)
        out = s0_roi(out, bboxes)
        out = self.s0_spool(out)
        pool_out.append(out)

        out = s1_tpool(inputs[1], 2)
        out = s1_roi(out, bboxes)
        out = self.s1_spool(out)
        pool_out.append(out)

        # B C H W.
        cat_ = ops.Concat(1)
        x = cat_(pool_out)

        # Perform dropout.
        if self.dropout_rate > 0.0:
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.act(x)
        return x
