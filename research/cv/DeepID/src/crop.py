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
"""Crop for DeepID Dataset"""

import os
from PIL import Image


def crop_img_by_half_center(src_file_path, dest_file_path):
    "crop img by half center"
    im = Image.open(src_file_path)
    x_size, _ = im.size
    start_point_xy = x_size / 4
    end_point_xy = x_size / 4 + x_size / 2
    box = (start_point_xy, start_point_xy, end_point_xy, end_point_xy)
    new_im = im.crop(box)
    new_new_im = new_im.resize((47, 55))
    new_new_im.save(dest_file_path)


def walk_through_the_folder_for_crop(aligned_db_folder, result_folder):
    "walk through the folder for crop"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    i = 0
    img_count = 0
    for people_folder in os.listdir(aligned_db_folder):
        src_people_path = aligned_db_folder + people_folder + '/'
        dest_people_path = result_folder + people_folder + '/'
        if not os.path.exists(dest_people_path):
            os.mkdir(dest_people_path)
        for video_folder in os.listdir(src_people_path):
            src_video_path = src_people_path + video_folder + '/'
            dest_video_path = dest_people_path + video_folder + '/'
            if not os.path.exists(dest_video_path):
                os.mkdir(dest_video_path)
            for img_file in os.listdir(src_video_path):
                src_img_path = src_video_path + img_file
                dest_img_path = dest_video_path + img_file
                crop_img_by_half_center(src_img_path, dest_img_path)
            i += 1
            img_count += len(os.listdir(src_video_path))
        print('Finish people:', people_folder)


if __name__ == '__main__':
    aligned_folder = "../data/aligned_images_DB"
    result_f = "../data/crop_images_DB"
    if not aligned_folder.endswith('/'):
        aligned_folder += '/'
    if not result_f.endswith('/'):
        result_f += '/'
    walk_through_the_folder_for_crop(aligned_folder, result_f)
