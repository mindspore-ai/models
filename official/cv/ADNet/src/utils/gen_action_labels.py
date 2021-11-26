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
# generate action labels for training the network
# matlab code:
# https://github.com/hellbell/ADNet/blob/master/utils/gen_action_labels.m

import numpy as np

from src.utils.overlap_ratio import overlap_ratio


def gen_action_labels(num_actions, opts, bb_samples, gt_bbox):
    num_samples = len(bb_samples)

    action_labels = np.zeros([num_actions, num_samples])
    m = opts['action_move']

    for j in range(len(bb_samples)):
        bbox = bb_samples[j, :]

        bbox[0] = bbox[0] + 0.5*bbox[2]
        bbox[1] = bbox[1] + 0.5*bbox[3]

        deltas = [m['x'] * bbox[2], m['y'] * bbox[3], m['w'] * bbox[2], m['h'] * bbox[3]]
        ar = bbox[2]/bbox[3]
        if bbox[2] > bbox[3]:
            deltas[3] = deltas[2] / ar
        else:
            deltas[2] = deltas[3] * ar

        deltas = np.tile(deltas, (num_actions, 1))
        action_deltas = np.multiply(m['deltas'], deltas)
        action_boxes = np.tile(bbox, (num_actions, 1))
        action_boxes = action_boxes + action_deltas
        action_boxes[:, 0] = action_boxes[:, 0] - 0.5 * action_boxes[:, 2]
        action_boxes[:, 1] = action_boxes[:, 1] - 0.5 * action_boxes[:, 3]

        overs = overlap_ratio(action_boxes, np.tile(gt_bbox, (num_actions, 1)))
        max_action = np.argmax(overs[:-2])   # translation overlap
        max_value = overs[max_action]

        if overs[opts['stop_action']] > opts['stopIou']:
            max_action = opts['stop_action']

        if max_value == overs[opts['stop_action']]:
            max_action = np.argmax(overs[:])  # (trans + scale) action

        action = np.zeros(num_actions)
        action[max_action] = 1
        action_labels[:, j] = action

        # return bbox back
        bbox[0] = bbox[0] - 0.5 * bbox[2]
        bbox[1] = bbox[1] - 0.5 * bbox[3]

    return action_labels  # in real matlab code, they also return overs
