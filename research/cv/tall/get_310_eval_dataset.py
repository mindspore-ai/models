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
Use this file to generate ascend310 inference datasets
"""

import argparse
import os
import numpy as np
from src.dataset import TestingDataSet

def get_args():
    """ get args"""
    parser = argparse.ArgumentParser(description='Train CTRL')
    parser.add_argument('--eval_data_dir', type=str, default="/home/yuanyibo/dataset",
                        help='the directory of train data.')
    return parser.parse_args()

args = get_args()

if __name__ == '__main__':
    img_path = os.path.join(args.eval_data_dir, "Interval128_256_overlap0.8_c3d_fc6/")
    csv_path = os.path.join(args.eval_data_dir, "exp_data/TACoS/test_clip-sentvec.pkl")
    dataset = TestingDataSet(img_path, csv_path, 1)
    target_path = './310_eval_dataset'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for movie_name in dataset.movie_names:
        batch = 128
        print("loading movie data: " + movie_name + "....")
        movie_clip_featmaps, movie_clip_sentences = dataset.load_movie_slidingclip(movie_name, 16)
        for k in range(0, len(movie_clip_sentences), batch):
            sent_vec = [x[1] for x in movie_clip_sentences[k:k + batch]]
            length_k = len(sent_vec)
            sent_vec = np.array(sent_vec)
            if length_k < batch:
                padding = np.zeros(shape=[batch - length_k, sent_vec.shape[1]], dtype=np.float32)
                sent_vec = np.concatenate((sent_vec, padding), axis=0)
            batch = 128
            for t in range(0, len(movie_clip_featmaps), batch):
                featmap = [x[1] for x in movie_clip_featmaps[t:t + batch]]
                length_t = len(featmap)
                featmap = np.array(featmap)
                visual_clip_name = [x[0] for x in movie_clip_featmaps[t:t + batch]]
                if length_t < batch:
                    padding = np.zeros(shape=[batch - length_t, featmap.shape[1]], dtype=np.float32)
                    featmap = np.concatenate((featmap, padding), axis=0)
                start = np.array([int(x.split("_")[1]) for x in visual_clip_name])
                end = np.array([int(x.split("_")[2].split("_")[0]) for x in visual_clip_name])
                input_feat = np.concatenate((featmap, sent_vec), axis=1)
                name = movie_name+'_sent_'+str(k)+'_clip_'+str(t)
                input_feat.tofile(os.path.join(target_path, f'{name}.data'))
