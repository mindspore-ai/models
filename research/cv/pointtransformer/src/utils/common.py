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

import numpy as np
import mindspore as ms
from mindspore import context, nn
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, init, get_group_size

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def exists(val):
    return val is not None


def farthest_point_sample(point, npoint):
    N, _ = point.shape
    xyz = point[:, :3]
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class shift_point_cloud:
    """
    Randomly shift point cloud. Shift is per point cloud.
    """
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, batch_pc):
        _, C = batch_pc.shape

        shifts = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
        if C > 3:
            batch_pc[..., 0:3] += shifts
        else:
            batch_pc += shifts
        return batch_pc


class random_scale_point_cloud:
    """
    Randomly scale the point cloud. Scale is per point cloud.
    """
    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, batch_pc):
        _, C = batch_pc.shape
        scales = np.random.uniform(self.scale_low, self.scale_high)
        if C > 3:
            batch_pc[..., 0:3] *= scales
        else:
            batch_pc *= scales

        return batch_pc


class random_point_dropout:
    """
    Pad the boxes and labels.
    """
    def __init__(self, max_dropout_ratio=0.8):
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, batch_pc):
        """
        Call method.
        """
        dropout_ratio = np.random.random()*self.max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[0])) <= dropout_ratio)[0]
        if drop_idx.size != 0:
            batch_pc[drop_idx, :] = batch_pc[0, :] # set to the first point

        return batch_pc


def context_device_init(config, mode=context.GRAPH_MODE):
    config.rank_id = 0
    config.rank_size = 1
    if config.device_target in ["Ascend"]:
        context.set_context(mode=mode, device_target=config.device_target, save_graphs=False)
        if config.run_distribute:
            init()
            context.set_auto_parallel_context(
                parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            config.rank_id = get_rank()
            config.rank_size = get_group_size()
            context.set_auto_parallel_context(device_num=config.rank_size)
    else:
        raise ValueError("Only support Ascend.")


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._log_softmax = nn.LogSoftmax()

    def construct(self, data, _, label2):
        output = self._backbone(data)
        pred = self._log_softmax(output)
        _, _, C = pred.shape
        pred = pred.view(-1, C)
        label2 = label2.view(-1, 1)[:, 0]
        weight = ms.Tensor(np.ones((C)), ms.float32)
        loss, weight = self._loss_fn(pred, label2, weight)
        return loss
