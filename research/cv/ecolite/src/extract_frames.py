# -*- coding:UTF-8 -*-
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

"""
extract video frames.
"""
import os
import cv2
import argparse


def get_args():
    """ get args"""
    parser = argparse.ArgumentParser(description='extract video frames.')
    parser.add_argument('--video_src_path', type=str, default=None, help='the directory of train data.')
    parser.add_argument('--image_save_path', type=str, default=None, help='the directory of train check_point.')
    return parser.parse_args()


args = get_args()

videos_src_path = args.video_src_path
videos_save_path = args.image_save_path
videodirs = os.listdir(videos_src_path)
prefix = 'img_'
for each_video_dir in videodirs:

    videos = os.listdir(videos_src_path + '/' + each_video_dir)

    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        os.makedirs(videos_save_path + '/' + each_video_name)
        each_video_save_full_path = os.path.join(videos_save_path + '/' + each_video_name) + '/'
        each_video_full_path = os.path.join(videos_src_path + '/' + each_video_dir + '/' + each_video)
        cap = cv2.VideoCapture(each_video_full_path)

        frame_count = 1
        success = True
        while success:
            success, frame = cap.read()
            print(success)
            if success:
                cv2.imwrite(each_video_save_full_path + prefix + "%06d.jpg" % frame_count, frame)
            frame_count = frame_count + 1
