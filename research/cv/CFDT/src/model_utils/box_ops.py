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
"""box ops"""
import numpy as np



def box_xyxy_to_cxcywh(x):
    """box xyxy to cxcywh"""
    x0, y0, x1, y1 = np.array_split(x.T, 4)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)[0]


def box_cxcywh_to_xyxy(x):
    """box cxcywh to xyxy"""
    x_c, y_c, w, h = np.array_split(x, 4, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1).squeeze(-2)


def box_xyxy_to_xywh(boxes):
    """box xyxy to xywh"""
    xmin, ymin, xmax, ymax = np.array_split(boxes.T, 4)
    return np.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=-1)[0]
