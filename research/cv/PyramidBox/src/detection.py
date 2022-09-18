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
from mindspore import Tensor
from src.bbox_utils import decode, nms

class Detect:
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE

    def detect(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (Tensor) Loc preds from loc layers
                Shape: [batch, num_priors*4]
            conf_data: (Tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors, num_classes]
            prior_data: Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        if isinstance(loc_data, Tensor):
            loc_data = loc_data.asnumpy()
        if isinstance(conf_data, Tensor):
            conf_data = conf_data.asnumpy()

        num = loc_data.shape[0]
        num_priors = prior_data.shape[0]

        conf_preds = np.transpose(conf_data.reshape((num, num_priors, self.num_classes)), (0, 2, 1))
        batch_priors = prior_data.reshape((-1, num_priors, 4))
        batch_priors = np.broadcast_to(batch_priors, (num, num_priors, 4))
        decoded_boxes = decode(loc_data.reshape((-1, 4)), batch_priors, self.variance).reshape((num, num_priors, 4))

        output = np.zeros((num, self.num_classes, self.top_k, 5))

        for i in range(num):
            boxes = decoded_boxes[i].copy()
            conf_scores = conf_preds[i].copy()

            for cl in range(1, self.num_classes):
                c_mask = np.greater(conf_scores[cl], self.conf_thresh)
                scores = conf_scores[cl][c_mask]

                if scores.ndim == 0:
                    continue

                l_mask = np.expand_dims(c_mask, 1)
                l_mask = np.broadcast_to(l_mask, boxes.shape)

                boxes_ = boxes[l_mask].reshape((-1, 4))

                ids, count = nms(boxes_, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = np.concatenate((np.expand_dims(scores[ids[:count]], 1),
                                                        boxes_[ids[:count]]), 1)
        return output
