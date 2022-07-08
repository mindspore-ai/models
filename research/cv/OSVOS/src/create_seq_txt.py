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
# ===========================================================================
"""
create sequence name file for training.
"""
import os
import argparse


parser = argparse.ArgumentParser(description='create sequence name file.')
parser.add_argument("--out_path", type=str, default='./DAVIS', help="the sequence name files output path.")

if __name__ == "__main__":
    args = parser.parse_args()
    seq_name_train_list = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump',
                           'dog-agility', 'drift-turn', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low',
                           'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike',
                           'paragliding', 'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 'stroller',
                           'surf', 'swing', 'tennis', 'train']
    seq_name_val_list = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows',
                         'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf',
                         'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']
    seq_name_train_txt = os.path.join(args.out_path, 'train.txt')
    seq_name_val_txt = os.path.join(args.out_path, 'val.txt')
    with open(seq_name_train_txt, 'w') as f1:
        for name in seq_name_train_list:
            f1.writelines(name + '\n')
    with open(seq_name_val_txt, 'w') as f2:
        for name in seq_name_val_list:
            f2.writelines(name + '\n')
    f1.close()
    f2.close()
