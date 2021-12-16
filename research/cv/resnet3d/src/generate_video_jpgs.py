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
Convert avi to jpg.
"""
import os
import argparse
import cv2


def generate_video_jpgs(args_opt):
    """
    Convert video to jpg.
    """
    video_src_src_path = args_opt.video_path
    video_src_TAG_path = args_opt.target_path

    label_name = os.listdir(video_src_src_path)
    total_class = len(label_name)
    label_dir = {}
    index = 0
    for i in label_name:
        if i.startswith('.'):
            continue
        label_dir[i] = index
        index += 1
        video_src_path = os.path.join(video_src_src_path, i)
        # video_save_path = os.path.join(video_src_TAG_path, i) + '_jpg'
        video_save_path = os.path.join(video_src_TAG_path, i)
        if not os.path.exists(video_src_TAG_path):
            os.mkdir(video_src_TAG_path)
        if not os.path.exists(video_save_path):
            os.mkdir(video_save_path)

        videos = os.listdir(video_src_path)
        total_video_per_calss = len(videos)
        videos = filter(lambda x: x.endswith('avi'), videos)
        index_video = 0
        for each_video in videos:
            each_video_name, _ = each_video.split('.')
            if not os.path.exists(video_save_path + '/' + each_video_name):
                os.mkdir(video_save_path + '/' + each_video_name)

            each_video_save_full_path = os.path.join(
                video_save_path, each_video_name) + '/'

            each_video_full_path = os.path.join(video_src_path, each_video)
            print("handling class {} / {} \nvideo of class {} / {}".format(index, total_class,
                                                                           index_video, total_video_per_calss),
                  each_video_full_path)
            index_video += 1
            cap = cv2.VideoCapture(each_video_full_path)
            frame_count = 1
            success = True
            while success:
                success, frame = cap.read()
                # print('read a new frame:', success)

                params = []
                params.append(1)
                if success:
                    # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame, params)
                    cv2.imwrite(each_video_save_full_path + "image_%05d.jpg" % frame_count, frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 75])

                frame_count += 1
            cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='src video path')
    parser.add_argument('--target_path', type=str, help='target jpg path')
    args = parser.parse_args()
    generate_video_jpgs(args)
