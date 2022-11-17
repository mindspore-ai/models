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
# Get video information (image paths and ground truths)
# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/get_video_infos.m

import os
import glob

ROOT_PATH = os.path.dirname(__file__)
def get_video_infos(bench_name, video_path, video_name):
    assert bench_name in ['vot13', 'vot14', 'vot15']

    if bench_name in ['vot13', 'vot14', 'vot15']:
        # path to VOT dataset
        video_info = {
            'gt': [],
            'img_files': [],
            'name': video_name,
            'db_name': bench_name,
            'nframes': 0
        }
        benchmarkSeqHome = video_path
        # img path
        imgDir = os.path.join(benchmarkSeqHome, video_name)
        if not os.path.exists(imgDir):
            print(imgDir + ' does not exist!')
            raise FileNotFoundError
        if bench_name == 'vot15':
            img_files = glob.glob(os.path.join(imgDir, 'color/*.jpg'))
        else:
            img_files = glob.glob(os.path.join(imgDir, '*.jpg'))
        img_files.sort(key=str.lower)

        for i in range(len(img_files)):
            img_path = os.path.join(img_files[i])
            video_info['img_files'].append(img_path)

        # gt path
        gtPath = os.path.join(benchmarkSeqHome, video_name, 'groundtruth.txt')
        if not os.path.exists(gtPath):
            print(gtPath + ' does not exist!')
            raise FileNotFoundError

        # parse gt
        gtFile = open(gtPath, 'r')
        gt = gtFile.read().split('\n')
        for i in range(len(gt)):
            if gt[i] == '' or gt[i] is None:
                continue
            gt[i] = gt[i].split(',')
            gt[i] = list(map(float, gt[i]))
        gtFile.close()

        if len(gt[0]) >= 6:
            for gtidx in range(len(gt)):
                if gt[gtidx] == "":
                    continue
                x = gt[gtidx][0:len(gt[gtidx]):2]
                y = gt[gtidx][1:len(gt[gtidx]):2]
                gt[gtidx] = [min(x),
                             min(y),
                             max(x) - min(x),
                             max(y) - min(y)]

        video_info['gt'] = gt

        video_info['nframes'] = min(len(video_info['img_files']), len(video_info['gt']))
        video_info['img_files'] = video_info['img_files'][:video_info['nframes']]
        video_info['gt'] = video_info['gt'][:video_info['nframes']]

        return video_info
    return {}
