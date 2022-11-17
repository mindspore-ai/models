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
import os

import numpy as np
import cv2

from src.utils.display import draw_box


def draw_box_from_npy(video_path, npy_file, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    bboxes = np.load(npy_file)

    frames_files = os.listdir(video_path)
    frames_files.sort(key=str.lower)

    for frame_idx, frame_file in enumerate(frames_files):
        frame = cv2.imread(os.path.join(video_path, frame_file))
        curr_bbox = bboxes[frame_idx]
        im_with_bb = draw_box(frame, curr_bbox)

        filename = os.path.join(save_path, str(frame_idx) + '.jpg')
        cv2.imwrite(filename, im_with_bb)
