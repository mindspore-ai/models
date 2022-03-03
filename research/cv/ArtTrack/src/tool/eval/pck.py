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
from numpy import array as arr


def enclosing_rect(points):
    """
    enclose rectangle
    """
    xs = points[:, 0]
    ys = points[:, 1]
    return np.array([np.amin(xs), np.amin(ys), np.amax(xs), np.amax(ys)])


def rect_size(rect):
    """
    get rectangle size
    """
    return np.array([rect[2] - rect[0], rect[3] - rect[1]])


def print_results(pck, cfg):
    """
    print result
    """
    _str = ""
    for heading in cfg.all_joints_names + ["total"]:
        _str += " & " + heading
    print(_str)

    _str = ""
    all_joint_ids = cfg.all_joints + [np.arange(cfg.num_joints)]
    for j_ids in all_joint_ids:
        j_ids_np = arr(j_ids)
        pck_av = np.mean(pck[j_ids_np])
        _str += " & {0:.1f}".format(pck_av)
    print(_str)
