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

from src.utils import generate_box, nms, convert_to_square, pad, calibrate_box, process_image
from src.models.predict_nets import predict_pnet, predict_rnet, predict_onet
import config as cfg

import numpy as np
import cv2


class MtcnnDetector:
    """Detect Image By MTCNN Model"""
    def __init__(self, pnet, rnet, onet, min_face_size=20, scale_factor=0.79):
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor

    def detect_pnet(self, im, thresh=cfg.P_THRESH):
        """Get face candidates through pnet
        Parameters:
        ----------
        im: numpy array
            input image array
        Returns:
        -------
        boxes_c: numpy array
            boxes after calibration
        """
        net_size = 12
        current_scale = float(net_size) / self.min_face_size
        im_resized = process_image(im, current_scale)
        _, current_height, current_width = im_resized.shape
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            cls_map, reg = predict_pnet(im_resized, self.pnet)
            boxes = generate_box(cls_map[1, :, :], reg, current_scale, thresh)

            current_scale *= self.scale_factor
            im_resized = process_image(im, current_scale)
            _, current_height, current_width = im_resized.shape

            if boxes.size == 0:
                continue
            keep = nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if not all_boxes:
            return None

        all_boxes = np.vstack(all_boxes)

        keep = nms(all_boxes[:, 0:5], 0.7)
        all_boxes = all_boxes[keep]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T
        return boxes_c

    def detect_rnet(self, im, dets, thresh=cfg.R_THRESH):
        """Get face candidates using rnet
        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet
        Returns:
        -------
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, _ = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        delete_size = np.ones_like(tmpw) * 20
        ones = np.ones_like(tmpw)
        zeros = np.zeros_like(tmpw)
        num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
        cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        for i in range(int(num_boxes)):
            if tmph[i] < 20 or tmpw[i] < 20:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            try:
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
                img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
                img = img.transpose((2, 0, 1))
                img = (img - 127.5) / 128
                cropped_ims[i, :, :, :] = img
            except ValueError:
                continue
        cls_scores, reg = predict_rnet(cropped_ims, self.rnet)
        if cls_scores.ndim < 2:
            cls_scores = cls_scores[None, :]
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > thresh)[0]
        if keep_inds.size != 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None
        keep = nms(boxes, 0.4)
        boxes = boxes[keep]

        boxes_c = calibrate_box(boxes, reg[keep])
        return boxes_c

    def detect_onet(self, im, dets, thresh=cfg.O_THRESH):
        """Get face candidates using onet
        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet
        Returns:
        -------
        boxes_c: numpy array
            boxes after calibration
        landmark: numpy array
        """
        h, w, _ = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img

        cls_scores, reg, landmark = predict_onet(cropped_ims, self.onet)
        if cls_scores.ndim < 2:
            cls_scores = cls_scores[None, :]
        if reg.ndim < 2:
            reg = reg[None, :]
        if landmark.ndim < 2:
            landmark = landmark[None, :]

        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > thresh)[0]
        if keep_inds.size != 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        w = boxes[:, 2] - boxes[:, 0] + 1

        h = boxes[:, 3] - boxes[:, 1] + 1

        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = calibrate_box(boxes, reg)

        keep = nms(boxes_c, 0.6, mode='Minimum')
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes_c, landmark

    def detect_face(self, image_path):
        im = cv2.imread(image_path)

        boxes_c = self.detect_pnet(im, 0.9)
        if boxes_c is None:
            return None, None

        boxes_c = self.detect_rnet(im, boxes_c, 0.6)
        if boxes_c is None:
            return None, None

        boxes_c, landmark = self.detect_onet(im, boxes_c, 0.7)
        if boxes_c is None:
            return None, None

        return boxes_c, landmark
