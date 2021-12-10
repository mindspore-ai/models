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
"""convert videos to images"""

import os
import threading

NUM_THREADS = 100
VIDEO_ROOT = '/ssd/video/something/v2/20bn-something-something-v2'         # Downloaded webm videos
FRAME_ROOT = '/ssd/video/something/v2/20bn-something-something-v2-frames'  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for k in range(0, len(l), n):
        yield l[k:k + n]


def extract(video, tmpl='%06d.jpg'):
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(VIDEO_ROOT, video,
                                                                                             FRAME_ROOT, video[:-5])
    os.system(cmd)


def target(videos_list):
    for video in videos_list:
        os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))
        extract(video)


if __name__ == '__main__':
    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)

    video_list = os.listdir(VIDEO_ROOT)
    splits = list(split(video_list, NUM_THREADS))

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
