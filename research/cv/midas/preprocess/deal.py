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
"""preprocess Kitti"""

import os
import argparse

def deal(args):
    '''deal Kitti dataset'''

    txt_path = args.txt_path

    if not os.path.exists('../Kitti'):
        os.mkdir('../Kitti')

    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            all_path = line.split(' ')
            val_path = os.path.join(args.depth_path, 'val', all_path[0], 'proj_depth',
                                    'groundtruth', 'image_02', all_path[1] + '.png')
            target_path = '../Kitti/' + all_path[0]
            if not os.path.exists(target_path):
                os.mkdir(target_path)
                os.mkdir(target_path + '/depth')
                os.mkdir(target_path + '/image')
            date = all_path[0].split('_drive')[0]
            image_path = os.path.join(args.image_path, date, all_path[0], 'image_02', 'data', all_path[1] + '.png')
            print(image_path)
            cp_command = 'cp ' + val_path + ' ../Kitti/' + all_path[0] + '/depth'
            cp1_command = 'cp ' + image_path + ' ../Kitti/' + all_path[0] + '/image'
            #print(cp_command)
            os.system(cp_command)
            os.system(cp1_command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deal with Kitti dataset')
    parser.add_argument('--depth_path', default='./data_depth_annotated')
    parser.add_argument('--txt_path', default='./write_file.txt')
    parser.add_argument('--image_path', default='./')
    args1 = parser.parse_args()
    deal(args1)
