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
"""common utils"""
import math
from typing import Union
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore as ms
import mindspore.nn as nn
from mindspore.common import dtype as mstype
import mindspore.common.initializer as weight_init
import numpy as np
import cv2
from src.nms import batched_nms

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")
grad_scale = ms.ops.MultitypeFuncGraph("grad_scale")

@grad_scale.register("Tensor", "Tensor")
def gradient_scale(scale, grad):
    return grad * ms.ops.cast(scale, ms.ops.dtype(grad))


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class EfficientDetTrainOneStepCell(nn.TrainOneStepCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        enable_clip_grad (boolean): If True, clip gradients in BertTrainOneStepCell. Default: True.
    """

    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True):
        super(EfficientDetTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()
        self.enable_clip_grad = enable_clip_grad

    def set_sens(self, value):
        self.sens = value

    def construct(self, x, y):
        """Defines the computation performed."""
        weights = self.weights

        loss = self.network(x, y)
        grads = self.grad(self.network, weights)(x, y, self.cast(F.tuple_to_array((self.sens,)), mstype.float32))
        if self.enable_clip_grad:
            grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)

def _calculate_fan_in_and_fan_out(tensor):
    """
    _calculate_fan_in_and_fan_out
    """
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor"
                         " with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def init_weights(model):
    """ init weights of net"""

    for name, cell in model.cells_and_names():
        is_conv_layer = isinstance(cell, nn.Conv2d)

        if is_conv_layer:

            if "conv_list" in name or "header" in name:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                sigma = math.sqrt(1. / float(fan_in))
                data = ms.Tensor(np.random.normal(loc=0, scale=sigma, size=cell.weight.shape).astype(np.float32))
                cell.weight.set_data(weight_init.initializer(data, cell.weight.shape))

            else:
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

            if cell.has_bias is True:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    cell.bias.set_data(weight_init.initializer(bias_value, cell.bias.shape))
                else:
                    cell.bias.set_data(weight_init.initializer('zeros', cell.bias.shape))


def bbox_transform(anchors, regression):
    """ convert box x1y1x2y2 to xywh """
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = np.exp(regression[..., 3]) * wa
    h = np.exp(regression[..., 2]) * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    return np.stack([xmin, ymin, xmax, ymax], axis=2)


def clipBoxes(boxes, img):
    """ clip the overflow value """
    _, _, height, width = img.shape

    boxes[:, :, 0] = np.clip(boxes[:, :, 0], a_min=0, a_max=None)
    boxes[:, :, 1] = np.clip(boxes[:, :, 1], a_min=0, a_max=None)
    boxes[:, :, 2] = np.clip(boxes[:, :, 2], a_min=None, a_max=width - 1)
    boxes[:, :, 3] = np.clip(boxes[:, :, 3], a_min=None, a_max=height - 1)

    return boxes

def invert_affine(metas: Union[float, list, tuple], preds):
    """ resize the output to real size """
    for i in range(len(preds)):
        if preds[i]['rois'].shape[0] == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, _, _ = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    """ resize pad used for eval"""
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w
    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image
    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h


def preprocess(image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """preprocess"""
    ori_imgs = [cv2.imread(image_path)]

    normalized_imgs = [(img[:, :, ::-1] / 255 - mean) / std for img in ori_imgs]

    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]

    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]
    return ori_imgs, framed_imgs, framed_metas

def preprocess_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = frame_from_video
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

def postprocess(x, anchors, regression, classification, threshold, iou_threshold):
    """postprocess"""

    transformed_anchors = bbox_transform(anchors.asnumpy(), regression.asnumpy())
    transformed_anchors = clipBoxes(transformed_anchors, x.asnumpy())

    classification = classification.asnumpy()
    transformed_anchors = transformed_anchors # [1, 49104, 4]

    scores = np.max(classification, axis=2, keepdims=True)   # [1,49104,1]
    scores_over_thresh = (scores > threshold)[:, :, 0]     # [1,49104]

    out = []

    for i in range(x.shape[0]):     # 1

        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], :]   # (X,90)

        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], :]   # (X,4)

        scores_per = scores[i, scores_over_thresh[i, :], :]    # (X, 1)

        # nms筛选
        classes_ = np.argmax(classification_per, axis=1)   # (X)
        scores_ = np.amax(classification_per, axis=1)  # (X)

        # (x,4), (x), 0.5
        # anchors_nms_idx = _diou_nms(transformed_anchors_per, scores_per[:, 0], iou_threshold)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]
            out.append({
                'rois': boxes_,
                'class_ids': classes_,
                'scores': scores_,
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out

def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index

def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    """ plot box """
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
