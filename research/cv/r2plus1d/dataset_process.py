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
Prepare dataset.
split the dataset into 'train' and 'val',
then transform each video into images correspondingly.
if your dataset has already been splited into 'train' and 'val' (e.g. kinetics400),
just set the 'splited=1' in default_config.yaml
"""
import os
import logging
import random
import math
import shutil
import cv2
from decord import VideoReader, cpu
from decord._ffi.base import DECORDError
from src.config import config as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main_preprocess(source_data_dir, splited, output_dir):
    if not splited:
        all_video = []
        categorys = sorted(os.listdir(source_data_dir))
        for category in categorys:
            videos = sorted(os.listdir(os.path.join(source_data_dir, category)))
            all_video.extend([os.path.join(source_data_dir, category, video) for video in videos])
        random.seed(len(all_video))
        random.shuffle(all_video)
        os.makedirs(os.path.join(source_data_dir, "train"))
        os.makedirs(os.path.join(source_data_dir, "val"))
        train_set = all_video[:math.floor(len(all_video)*0.8)]
        val_set = all_video[math.floor(len(all_video)*0.8):]
        move_videos(train_set, source_data_dir, "train")
        move_videos(val_set, source_data_dir, "val")
        for category in categorys:
            shutil.rmtree(os.path.join(source_data_dir, category))
    video_category = sorted(os.listdir(source_data_dir))
    assert len(video_category) == 2 and video_category[0] == "train" and video_category[1] == "val", \
    "The dataset has not been splited correctly, please check it!!!"
    preprocess(os.path.join(source_data_dir, "train"), os.path.join(output_dir, "train"))
    preprocess(os.path.join(source_data_dir, "val"), os.path.join(output_dir, "val"))

def move_videos(video_set, source_data_dir, split_class):
    '''
    split_class: "train" or "val"
    '''
    assert split_class in ["train", "val"]
    for index, video in enumerate(video_set):
        if (index + 1) % 500 == 0:
            logger.info("Moving %s_set, %s/%s...", split_class, str(index+1), str(len(video_set)))
        category = video.split(os.sep)[-2]
        dest_dir = os.path.join(source_data_dir, split_class, category)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.move(video, dest_dir)

def preprocess(source_data_dir, output_dir, resize_height=128, resize_width=171):
    '''
    resize_height=128, resize_width=171 should not be changed
    '''
    logger.info("Start preprocessing...")
    for numm, file in enumerate(sorted(os.listdir(source_data_dir))):
        logger.info("video category:%s, num:%s", str(file), str(numm))
        file_path = os.path.join(source_data_dir, file)
        video_files = [name for name in os.listdir(file_path)]
        if not os.path.exists(os.path.join(output_dir, file)):
            os.makedirs(os.path.join(output_dir, file))
        for video in video_files:
            process_video(source_data_dir, video, file, os.path.join(output_dir, file), \
                          resize_height, resize_width)
    logger.info('Preprocessing finished.')

def process_video(root_dir, video, action_name, save_dir, resize_height, resize_width):
    '''
    transform single video into images
    '''
    video_filename = video.split('.')[0]
    if os.path.getsize(os.path.join(root_dir, action_name, video)) < 1 * 1024:
        logger.info("SKIP: %s - %s", str(os.path.join(root_dir, action_name, video)), \
                    str(os.path.getsize(os.path.join(root_dir, action_name, video))))
    try:
        vr = VideoReader(os.path.join(root_dir, action_name, video), width=resize_width, height=resize_height,
                         num_threads=1, ctx=cpu(0))
    except DECORDError as _:
        logger.info('Video decode failed:%s', str(os.path.join(root_dir, action_name, video)))
        return
    if not os.path.exists(os.path.join(save_dir, video_filename)):
        os.makedirs(os.path.join(save_dir, video_filename))
    for i in range(len(vr)):
        cv2.imwrite(os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), vr[i].asnumpy())

if __name__ == "__main__":
    main_preprocess(source_data_dir=cfg.source_data_dir,
                    splited=cfg.splited,
                    output_dir=cfg.output_dir)
