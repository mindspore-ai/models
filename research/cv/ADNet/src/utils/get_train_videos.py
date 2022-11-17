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
# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/get_train_videos.m

import numpy as np

from src.utils.get_benchmark_path import get_benchmark_path
from src.utils.get_benchmark_info import get_benchmark_info


def get_train_videos(opts, args):
    train_db_names = opts['train_dbs']
    test_db_names = opts['test_db']

    video_names = []
    video_paths = []
    bench_names = []

    for dbidx in range(len(train_db_names)):
        bench_name = train_db_names[dbidx]
        path_ = get_benchmark_path(bench_name, args)
        video_names_ = get_benchmark_info(train_db_names[dbidx] + '-' + test_db_names)
        video_paths_ = np.tile(path_, (1, len(video_names_)))
        video_names.extend(video_names_)
        video_paths.extend(list(video_paths_[0]))
        #np.tile(
        bench_names.extend(list(np.tile(bench_name, (1, len(video_names_)))[0]))

    train_db = {
        'video_names': video_names,
        'video_paths': video_paths,
        'bench_names': bench_names
    }
    return train_db
