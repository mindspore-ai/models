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
inference
"""

import json
import time
from collections import defaultdict
import numpy as np

from mindspore import Tensor

from .videodataset_multiclips import get_target_path


def topk_(matrix, K, axis=1):
    """
    Calculate topk result.
    """
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort


def naive_topK(matrix, K):
    """
    Calculate topk for a given matrix.
    """
    sorted_data = -np.sort(-matrix)
    sorted_data = sorted_data[0:K]
    sorted_idx = np.argsort(-matrix)
    sorted_idx = sorted_idx[0:K]
    return sorted_data, sorted_idx


def get_video_results(outputs, class_names, output_topk):
    """
    Get inference results for video.
    """
    K = min(output_topk, len(class_names))
    sorted_scores, locs = naive_topK(outputs, K)
    video_results = []
    for idx, score in enumerate(sorted_scores):
        video_results.append({
            'label': class_names[str(locs[idx])],
            'score': score.item()
        })
    return video_results


def get_class_labels(data):
    """
    Get labels for all classes.
    """
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def load_ground_truth(ground_truth_path, subset):
    """
    Load ground truth.
    """
    with ground_truth_path.open('r') as f:
        data = json.load(f)

    # Category name(str) <-> Category number(int)
    class_labels_map = get_class_labels(data)

    ground_truth = []
    for video_id, v in data['database'].items():
        if subset != v['subset']:
            continue
        this_label = v['annotations']['label']
        ground_truth.append((video_id, class_labels_map[this_label]))

    return ground_truth, class_labels_map


def load_result(data, class_labels_map):
    """
    Load inference results.
    """
    result_top1 = {}
    result_top5 = {}

    for video_id, v in data['results'].items():
        labels_and_scores = []
        for this_result in v:
            label = class_labels_map[this_result['label']]  # class number(int)
            score = this_result['score']
            labels_and_scores.append((label, score))
        labels_and_scores.sort(key=lambda x: x[1], reverse=True)
        result_top1[video_id] = list(zip(*labels_and_scores[:1]))[0]
        result_top5[video_id] = list(zip(*labels_and_scores[:5]))[0]

    return result_top1, result_top5


def calculate_clip_acc(clips_results, ground_truth, class_labels_map):
    """
    Calculate clip accuracy.
    """
    correct = 0
    total_clips = 0
    predict_results = clips_results['results']
    for (video_id, true_label) in ground_truth:
        video_result_list = predict_results[video_id]
        for per_clip_result in video_result_list:
            # class name -> class number
            predict_label = class_labels_map[per_clip_result['result'][0]['label']]
            correct += (1 if predict_label == true_label else 0)
            total_clips += 1
    clip_acc = float(correct) / float(total_clips)

    return clip_acc


def remove_nonexistent_ground_truth(ground_truth, result):
    """
    Remove non-existent fround ground truth.
    """
    exist_ground_truth = [line for line in ground_truth if line[0] in result]

    return exist_ground_truth


class Inference:
    """
    Inference.
    """

    def __init__(self):
        """
        Init class Inference.
        """

    def __call__(self, predict_data, model, annotation_path):
        total_target_path = get_target_path(annotation_path)
        with total_target_path.open('r') as f:
            total_target_data = json.load(f)
        results = {'results': defaultdict(list)}

        size = predict_data.get_dataset_size()
        for step, data in enumerate(predict_data.create_dict_iterator(output_numpy=True)):
            t1 = time.time()
            x, label = data['data'][0], data['label'].tolist()
            video_ids, segments = zip(
                *total_target_data['targets'][str(label[0])])
            x_list = np.split(x, x.shape[0], axis=0)
            outputs = []
            for x in x_list:
                output = model.predict(Tensor(x))
                outputs.append(output.asnumpy())
            outputs = np.concatenate(outputs, axis=0)

            _, locs = topk_(outputs, K=1)
            locs = locs.reshape(1, -1)

            t2 = time.time()
            print("[{} / {}] Net time: {} ms".format(step, size, (t2 - t1) * 1000))
            for j in range(0, outputs.shape[0]):
                results['results'][video_ids[j]].append({
                    'segment': segments[j],
                    'output': outputs[j]
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

        return inference_results, clips_inference_results
