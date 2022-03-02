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
Generate n_frames files for datasets.
"""

from __future__ import print_function, division

import os
import sys


def class_process(path, name):
    class_path = os.path.join(path, name)
    if not os.path.isdir(class_path):
        return

    for file_name in os.listdir(class_path):
        video_dir_path = os.path.join(class_path, file_name)
        image_indices = []
        for image_file_name in os.listdir(video_dir_path):
            if 'image' not in image_file_name:
                continue
            image_indices.append(int(image_file_name[6:11]))

        if image_indices:
            image_indices.sort(reverse=True)
            n_frames = image_indices[0]
            print(video_dir_path, n_frames)
        else:
            print('no image files', video_dir_path)
            n_frames = 0
        with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
            dst_file.write(str(n_frames))


if __name__ == "__main__":
    dir_path = sys.argv[1]
    for class_name in os.listdir(dir_path):
        class_process(dir_path, class_name)
