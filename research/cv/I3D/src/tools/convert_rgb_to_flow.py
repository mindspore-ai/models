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
"""
Convert RGB pictures to Flow type.
"""

import argparse
import os
from glob import glob

import cv2
import numpy as np


def compute_TVL1(prev, curr, bound=20):
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def frame_len_check(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()
    return len(frames)


def cal_for_frames(video_path):
    print(video_path)
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        if i == 1:
            print('Start reading')
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "image_{:05d}x.jpg".format(i + 1)), flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path.format('v'), "image_{:05d}y.jpg".format(i + 1)), flow[:, :, 1])


def extract_flow(rgb_path, flow_path):
    exception = []
    class_dirs = os.listdir(rgb_path)
    for class_dir in class_dirs:
        video_root = os.path.join(rgb_path, class_dir)
        video_dirs = os.listdir(video_root)
        for video in video_dirs:
            jpg_root = os.path.join(video_root, video)
            length = frame_len_check(jpg_root)
            if length == 0:
                print('File name exception')
                exception.append(jpg_root)
                continue
            else:
                save_root_part = jpg_root.replace(str(rgb_path), '').replace('\\', '/')
                save_path = flow_path + save_root_part
                if os.path.exists(save_path):
                    print('file exists:', save_path)
                    continue
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                flow = cal_for_frames(jpg_root)
                save_flow(flow, save_path)
                print('complete:' + save_path)

    print('#' * 60)
    print('File name exception:')
    for exc in exception:
        print(exc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_path', type=str, required=True,
                        help='This is the path where images sampled from the video are saved')
    parser.add_argument('--flow_path', type=str, required=True,
                        help='This is the path where you want save the flow files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()
    extract_flow(config.rgb_path, config.flow_path)
