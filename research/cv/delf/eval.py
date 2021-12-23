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
""" Eval mAp"""

import os
import argparse
import numpy as np

from src import dataset


parser = argparse.ArgumentParser(description='MindSpore delf eval')


parser.add_argument('--worker_size', type=int, default=16)
parser.add_argument('--ranks_path', type=str, default="")
parser.add_argument('--ranks_file', type=str, default="")
parser.add_argument('--image_path', type=str, default="")
parser.add_argument('--gt_path', type=str, default="")
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--metric_name', type=str, default="")

args = parser.parse_known_args()[0]

def main():
    query_list, index_list, ground_truth = dataset.read_ground_truth(
        args.gt_path, args.image_path)

    num_index_images = len(index_list)

    (easy_ground_truth, medium_ground_truth,
     hard_ground_truth) = dataset.ParseEasyMediumHardGroundTruth(ground_truth)


    ranks = np.zeros([0, num_index_images], dtype='int32')
    for i in range(args.worker_size):
        ranks_t = np.load(os.path.join(
            args.ranks_path, os.path.join('process'+str(i), args.ranks_file+str(i)+'.npz')))['arr_0']
        ranks = np.concatenate((ranks, ranks_t), axis=0)

    print('query_list')
    for i in range(len(query_list)):
        print(i, ": ", query_list[i])
    print('index_list')
    for i in range(len(index_list)):
        print(i, ": ", index_list[i])
    print(ground_truth)
    print(ranks)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    _PR_RANKS = range(num_index_images)[10:100:10]

    # Compute metrics.
    easy_metrics = dataset.ComputeMetrics(
        ranks, easy_ground_truth, _PR_RANKS)
    medium_metrics = dataset.ComputeMetrics(
        ranks, medium_ground_truth, _PR_RANKS)
    hard_metrics = dataset.ComputeMetrics(
        ranks, hard_ground_truth, _PR_RANKS)


    # Write metrics to file.
    mean_average_precision_dict = {
        'easy': easy_metrics[0],
        'medium': medium_metrics[0],
        'hard': hard_metrics[0]
    }
    mean_precisions_dict = {'easy': easy_metrics[1], 'medium': medium_metrics[1], 'hard': hard_metrics[1]}
    mean_recalls_dict = {'easy': easy_metrics[2], 'medium': medium_metrics[2], 'hard': hard_metrics[2]}

    dataset.SaveMetricsFile(mean_average_precision_dict, mean_precisions_dict,
                            mean_recalls_dict, _PR_RANKS,
                            os.path.join(args.output_dir, args.metric_name))


if __name__ == '__main__':
    main()
