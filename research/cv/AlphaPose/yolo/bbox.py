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
'''
bbox
'''
from __future__ import division


import random

import cv2
import mindspore
import mindspore.ops as ops
maximum = ops.Maximum()
minimum = ops.Minimum()
zeros = ops.Zeros()
def confidence_filter(result, confidence):
    '''
    confidence_filter_cls
    '''
    conf_mask = (result[:, :, 4] > confidence).float().unsqueeze(2)
    result = result*conf_mask

    return result

def confidence_filter_cls(result, confidence):
    '''
    confidence_filter_cls
    '''
    max_scores = torch.max(result[:, :, 5:25], 2)[0]
    res = torch.cat((result, max_scores), 2)
    print(res.shape)


    cond_1 = (res[:, :, 4] > confidence).float()
    cond_2 = (res[:, :, 25] > 0.995).float()

    conf = cond_1 + cond_2
    conf = torch.clamp(conf, 0.0, 1.0)
    conf = conf.unsqueeze(2)
    result = result*conf
    return result



def get_abs_coord(box):
    '''
    get_abs_coord
    '''
    box[2], box[3] = abs(box[2]), abs(box[3])
    x1 = (box[0] - box[2]/2) - 1
    y1 = (box[1] - box[3]/2) - 1
    x2 = (box[0] + box[2]/2) - 1
    y2 = (box[1] + box[3]/2) - 1
    return x1, y1, x2, y2



def sanity_fix(box):
    '''
    sanity_fix
    '''
    if box[0] > box[2]:
        box[0], box[2] = box[2], box[0]

    if box[1] > box[3]:
        box[1], box[3] = box[3], box[1]

    return box

def bbox_iou(box1, box2):
    """
    bbox_iou
    """

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]


    inter_rect_x1 = maximum(b1_x1, b2_x1)

    inter_rect_y1 = maximum(b1_y1, b2_y1)

    inter_rect_x2 = minimum(b1_x2, b2_x2)

    inter_rect_y2 = minimum(b1_y2, b2_y2)

    inter_area = maximum(inter_rect_x2 - inter_rect_x1 + 1,
                         zeros(inter_rect_x2.shape, mindspore.float32))* maximum(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                 zeros(inter_rect_x2.shape,
                                                                                       mindspore.float32))

    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def pred_corner_coord(prediction):
    '''
    pred_corner_coord
    '''
    ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()

    box = prediction[ind_nz[0], ind_nz[1]]


    box_a = box.new(box.shape)
    box_a[:, 0] = (box[:, 0] - box[:, 2]/2)
    box_a[:, 1] = (box[:, 1] - box[:, 3]/2)
    box_a[:, 2] = (box[:, 0] + box[:, 2]/2)
    box_a[:, 3] = (box[:, 1] + box[:, 3]/2)
    box[:, :4] = box_a[:, :4]

    prediction[ind_nz[0], ind_nz[1]] = box

    return prediction




def write(x, batches, results, colors, classes):
    '''
    write
    '''
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img
