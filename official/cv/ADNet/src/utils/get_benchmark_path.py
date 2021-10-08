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
# matlab source:
# https://github.com/hellbell/ADNet/blob/master/utils/get_benchmark_path.m
import os
import glob


def get_benchmark_path(bench_name, args):
    assert bench_name in ['vot15', 'vot14', 'vot13']
    if bench_name == 'vot15':
        video_path = glob.glob(os.path.join(args.dataset_path, '*15'))[0]
    elif bench_name == 'vot14':
        video_path = glob.glob(os.path.join(args.dataset_path, '*14'))[0]
    else:  # elif bench_name == 'vot13'
        video_path = glob.glob(os.path.join(args.dataset_path, '*13'))[0]

    return video_path
