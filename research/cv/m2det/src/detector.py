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

import mindspore
from mindspore import Tensor
from mindspore import nn
from mindspore import ops

from src.box_utils import decode


class Detect(nn.Cell):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.num_classes = num_classes
        self.variance = cfg['variance']
        self.zero_float = Tensor(0, dtype=mindspore.float32)

    def construct(self, predictions, prior):
        """
        Args:
            predictions: Predictions
            prior: Prior boxes and variances from priorbox layers
        """

        loc, conf = predictions

        bs = loc.shape[0]  # batch size
        priors_number = prior.shape[0]
        boxes = ops.BroadcastTo((1, priors_number, 4))(self.zero_float)
        scores = ops.BroadcastTo((1, priors_number, self.num_classes))(self.zero_float)

        if bs == 1:
            conf_preds = ops.ExpandDims()(conf, 0)  # Shape [bs, cls_num, priors_num]

        else:
            conf_preds = conf.view(bs, priors_number, self.num_classes)
            boxes = ops.BroadcastTo((bs, priors_number, 4))(boxes)
            scores = ops.BroadcastTo((bs, priors_number, self.num_classes))(scores)

        # Decode predictions into bounding boxes.
        for sample_index in range(bs):
            decoded_boxes = decode(loc[sample_index], prior, self.variance)
            conf_scores = conf_preds[sample_index].copy()

            boxes[sample_index] = decoded_boxes
            scores[sample_index] = conf_scores

        return boxes, scores
