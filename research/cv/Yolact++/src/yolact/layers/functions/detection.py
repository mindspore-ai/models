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
"""YOLACT++ clean up and full release."""
import mindspore
import mindspore.nn as nn
from mindspore import ops as P
from src.yolact.utils.ms_box_utils import jaccard
from src.config import yolact_plus_resnet50_config as cfg

class Detectx(nn.Cell):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        super().__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh

        self.use_cross_class_nms = False
        self.use_fast_nms = True
        self.x_max = P.ArgMaxWithValue(0)
        self.y_max = P.ArgMaxWithValue(1)
        self.less = P.Less()
        self.less_equal = P.LessEqual()
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.sort = P.TopK(sorted=True)
        self.tril = nn.Tril()
        self.range = nn.Range(80)
        self.expanddims = P.ExpandDims()
        shape = (80, 200)
        self.broadcast_to = P.BroadcastTo(shape)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.concat = P.Concat(1)
        self.exp = P.Exp()
        self.max_num_detections = cfg['max_num_detections']

    def construct(self, predictions, net):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]

        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        loc_data = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        prior_data = predictions['priors']
        prior_data = self.cast(prior_data, mindspore.float32)
        proto_data = predictions['proto'] # if 'proto' in predictions else None

        out = []

        batch_size = loc_data.shape[0]
        num_priors = prior_data.shape[0]

        perm = (0, 2, 1)
        x_conf_preds = self.reshape(conf_data, (batch_size, num_priors, self.num_classes))
        conf_preds = self.transpose(x_conf_preds, perm)


        for batch_idx in range(batch_size):
            decoded_boxes = self.decode(loc_data[batch_idx], prior_data)
            result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data)

            result['proto'] = proto_data[batch_idx]
            out.append({'detection': result})
        return out

    def decode(self, loc, priors):

        variances = [0.1, 0.2]
        boxes = self.concat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                             priors[:, 2:] * self.exp(loc[:, 2:] * variances[1])))
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes

    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        _, conf_scores = self.x_max(cur_scores)
        keep = self.less(self.conf_thresh, conf_scores)
        keep = self.cast(keep, mindspore.float32)
        scores = self.mul(cur_scores, keep)
        boxes = (self.mul((decoded_boxes).T, keep)).T
        masks = mask_data[batch_idx, :, :]
        masks = (self.mul((masks).T, keep)).T


        if scores.shape[1] == 0:
            return None

        boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def fast_nms(self, boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200,
                 second_threshold: bool = False):
        """fast nms"""

        scores, idx = self.sort(scores, top_k)

        num_classes, num_dets = idx.shape

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)
        x = self.tril(iou)
        iou = iou - x

        _, iou_max = self.y_max(iou)

        # Now just filter out the ones higher than the threshold
        keep = self.less_equal(iou_max, iou_threshold)
        keep = self.cast(keep, mindspore.int32)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.


        # Assign each kept detection to its corresponding class
        # classes = nn.Range(num_classes, device=boxes.device)[:, None].expand_as(keep)
        # classes = nn.Range(num_classes)[:, None].expand_as(keep)


        classes = self.broadcast_to(self.expanddims(self.range(), 1))

        classes = classes[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        scores = self.mul(scores, keep)
        scores = scores.view(-1)
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = self.sort(scores, self.max_num_detections)
        classes = classes.view(-1)
        boxes = boxes.view(-1, 4)
        masks = masks.view(-1, 32)
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores
