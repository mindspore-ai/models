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
Post process for 310 infer
"""
import os
import json
from collections import defaultdict
import numpy as np

from src.config import config as cfg
from src.videodataset_multiclips import get_target_path
from src.inference import (get_video_results, load_ground_truth, load_result,
                           remove_nonexistent_ground_truth, calculate_clip_acc)


if __name__ == '__main__':
    print(cfg)
    if cfg.n_classes == 101:
        dataset_name = 'ucf101'
    elif cfg.n_classes == 51:
        dataset_name = 'hmdb51'
    else:
        dataset_name = ''
    result_path = os.path.abspath(os.path.dirname(
        __file__)) + "/scripts/result_Files/"
    label_ids_path = os.path.abspath(os.path.dirname(
        __file__)) + "/scripts/preprocess_Result/label/label.json"

    total_target_path = get_target_path(cfg.annotation_path)
    with total_target_path.open('r') as f:
        total_target_data = json.load(f)

    results = {'results': defaultdict(list)}
    label_list = json.load(open(label_ids_path, 'r'))
    # print(label_list)
    for class_id, values in label_list.items():
        video_ids, segments = zip(
            *total_target_data['targets'][str(class_id)])
        for data in values:
            file_name = data['file_name']
            file_path = result_path + \
                file_name[:file_name.rfind('.')] + "_0.bin"
            idx = data['idx']
            output = np.fromfile(file_path, np.float32)
            results['results'][video_ids[idx]].append({
                'segment': segments[idx],
                'output': output
            })
    class_names = total_target_data['class_names']
    inference_results = {'results': {}}
    clips_inference_results = {'results': {}}
    for video_id, video_results in results['results'].items():
        video_outputs = [
            segment_result['output'] for segment_result in video_results
        ]
        video_outputs = np.stack(video_outputs, axis=0)
        average_scores = np.mean(video_outputs, axis=0)
        clips_inference_results['results'][video_id] = get_video_results(
            average_scores, class_names, 5)

        inference_results['results'][video_id] = []
        for segment_result in video_results:
            segment = segment_result['segment']
            result = get_video_results(segment_result['output'],
                                       class_names, 5)
            inference_results['results'][video_id].append({
                'segment': segment,
                'result': result
            })

    print('load ground truth')
    ground_truth, class_labels_map = load_ground_truth(
        cfg.annotation_path, "validation")
    print('number of ground truth: {}'.format(len(ground_truth)))

    n_ground_truth_top_1 = len(ground_truth)
    n_ground_truth_top_5 = len(ground_truth)

    result_top1, result_top5 = load_result(
        clips_inference_results, class_labels_map)

    ground_truth_top1 = remove_nonexistent_ground_truth(
        ground_truth, result_top1)
    ground_truth_top5 = remove_nonexistent_ground_truth(
        ground_truth, result_top5)

    if cfg.ignore:
        n_ground_truth_top_1 = len(ground_truth_top1)
        n_ground_truth_top_5 = len(ground_truth_top5)

    correct_top1 = [1 if line[1] in result_top1[line[0]]
                    else 0 for line in ground_truth_top1]
    correct_top5 = [1 if line[1] in result_top5[line[0]]
                    else 0 for line in ground_truth_top5]

    clip_acc = calculate_clip_acc(
        inference_results, ground_truth, class_labels_map)
    accuracy_top1 = float(sum(correct_top1)) / float(n_ground_truth_top_1)
    accuracy_top5 = float(sum(correct_top5)) / float(n_ground_truth_top_5)
    print('==================Accuracy=================\n'
          ' clip-acc : {} \ttop-1 : {} \ttop-5: {}'.format(clip_acc, accuracy_top1, accuracy_top5))
