# Copyright 2021 Huawei Technologies Co., Ltd
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

def do_action(bbox, opts, act, imSize):
    m = opts['action_move']

    # action
    bbox[0] = bbox[0] + 0.5 * bbox[2]
    bbox[1] = bbox[1] + 0.5 * bbox[3]

    deltas = [m['x'] * bbox[2],
              m['y'] * bbox[3],
              m['w'] * bbox[2],
              m['h'] * bbox[3]]

    deltas = np.maximum(deltas, 1)

    ar = bbox[2]/bbox[3]

    if bbox[2] > bbox[3]:
        deltas[3] = deltas[2] / ar

    else:
        deltas[2] = deltas[3] * ar

    action_delta = np.multiply(np.array(m['deltas'])[act, :], deltas)
    bbox_next = bbox + action_delta
    bbox_next[0] = bbox_next[0] - 0.5 * bbox_next[2]
    bbox_next[1] = bbox_next[1] - 0.5 * bbox_next[3]
    bbox_next[0] = np.maximum(bbox_next[0], 1)
    bbox_next[0] = np.minimum(bbox_next[0], imSize[1] - bbox_next[2])
    bbox_next[1] = np.maximum(bbox_next[1], 1)
    bbox_next[1] = np.minimum(bbox_next[1], imSize[0] - bbox_next[3])
    bbox_next[2] = np.maximum(5, np.minimum(imSize[1], bbox_next[2]))
    bbox_next[3] = np.maximum(5, np.minimum(imSize[0], bbox_next[3]))

    bbox[0] = bbox[0] - 0.5 * bbox[2]
    bbox[1] = bbox[1] - 0.5 * bbox[3]

    return bbox_next
