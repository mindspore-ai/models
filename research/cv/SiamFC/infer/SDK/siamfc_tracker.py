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
"""start eval (SDK)"""
import time
import numpy as np
import cv2
from tqdm import tqdm
from infer_utils_sdk import get_exemplar_image, get_exemplar, get_pyramid_instance_image, do_infer, show_image, config
from sdk_infer import SdkApi


class SiamFCSDKTracker:
    """
    The tracker for Siamfc
    """
    def __init__(self, pipeline1, pipeline2):
        self.pipeline1 = pipeline1
        self.pipeline2 = pipeline2
        self.name = 'SiamFC'

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))
                                                                 [np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """
        init this tracker
        """
        self.bbox = (bbox[0]-1, bbox[1]-1, bbox[0] -
                     1+bbox[2], bbox[1]-1 + bbox[3])
        self.cfg = config
        self.sdk_infer_exemplar = SdkApi(
            self.pipeline1, config.stream_name1)
        self.sdk_infer_exemplar.init()
        self.sdk_infer_instance = SdkApi(
            self.pipeline2, config.stream_name2)
        self.sdk_infer_instance.init()
        self.pos = np.array(
            [bbox[0]-1+(bbox[2]-1)/2, bbox[1]-1+(bbox[3]-1)/2])
        self.target_sz = np.array([bbox[2], bbox[3]])
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                                                        self.cfg.exemplar_size, self.cfg.context_amount,
                                                        self.img_mean)  # context_amount = 0.5
        exemplar_img = np.transpose(exemplar_img, (2, 0, 1))[None, :, :, :]
        exemplar_img = exemplar_img.astype(np.float32)
        none = np.ones(1)
        none = none.astype(np.float32)
        self.exemplar = get_exemplar(self.sdk_infer_exemplar,
                                     exemplar_img, none, self.cfg)
        self.penalty = np.ones((config.num_scale)) * config.scale_penalty
        self.penalty[config.num_scale // 2] = 1

        # create cosine window
        self.interp_response_sz = config.response_up_stride * config.response_sz
        self.cosine_window = self._cosine_window((self.interp_response_sz,
                                                  self.interp_response_sz))
        self.scales = config.scale_step**np.arange(np.ceil(3/2)-3,
                                                   np.floor(3/2)+1)  # [0.96385542,1,1.0375]
        self.s_x = s_z + (config.instance_size -
                          config.exemplar_size) / scale_z
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

    def update(self, frame):
        """
        updata the target position
        """
        size_x_scales = self.s_x * self.scales
        pyramid = get_pyramid_instance_image(frame, self.pos, config.instance_size,
                                             size_x_scales, self.img_mean)
        x_crops_tensor = ()
        for i in range(3):
            instance = np.transpose(pyramid[i], (2, 0, 1))[None, :, :, :]
            instance = instance.astype(np.float32)
            tmp_x_crop = do_infer(self.sdk_infer_instance,
                                  self.exemplar, instance, self.cfg)
            x_crops_tensor = x_crops_tensor+(tmp_x_crop,)
        response_maps = np.concatenate(x_crops_tensor, axis=0)
        response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz),
                                       cv2.INTER_CUBIC)for x in response_maps]
        max_score = np.array([x.max()
                              for x in response_maps_up]) * self.penalty
        scale_idx = max_score.argmax()
        response_map = response_maps_up[scale_idx]
        response_map -= response_map.min()
        response_map /= response_map.sum()
        response_map = (1 - config.window_influence) * response_map + \
            config.window_influence * self.cosine_window
        max_r, max_c = np.unravel_index(
            response_map.argmax(), response_map.shape)
        # displacement in interpolation response
        disp_response_interp = np.array(
            [max_c, max_r]) - (self.interp_response_sz - 1) / 2.
        # displacement in input
        disp_response_input = disp_response_interp * \
            config.total_stride/config.response_up_stride
        # displacement in frame
        scale = self.scales[scale_idx]
        disp_response_frame = disp_response_input * \
            (self.s_x * scale)/config.instance_size
        # position in frame coordinates
        self.pos += disp_response_frame
        # scale damping and saturation
        self.s_x *= ((1 - config.scale_lr) + config.scale_lr * scale)
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_lr) +
                          config.scale_lr * scale) * self.target_sz
        box = np.array([
            self.pos[0] + 1 - (self.target_sz[0]) / 2,
            self.pos[1] + 1 - (self.target_sz[1]) / 2,
            self.target_sz[0], self.target_sz[1]])
        return box

    def track(self, img_files, box, visualize=False):
        """
            To get the update track box and calculate time
            Args :
                img_files : the location of img
                box : the first image box, include x, y, width, high
        """
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box  # x，y, w, h
        times = np.zeros(frame_num)

        for f, img_file in tqdm(enumerate(img_files), total=len(img_files)):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin
            if visualize:
                show_image(img, boxes[f, :])
        return boxes, times
