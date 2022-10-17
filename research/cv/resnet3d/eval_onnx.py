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
# ============================================================================
"""
Eval.
"""
import time
import random
import json
from collections import defaultdict
import numpy as np
import onnxruntime
from mindspore import dataset as de
from mindspore.common import set_seed
from src.config import config as args_opt
from src.dataset import create_eval_dataset
from src.inference import (topk_, get_video_results, load_ground_truth, load_result,
                           remove_nonexistent_ground_truth, calculate_clip_acc)
from src.videodataset_multiclips import get_target_path


random.seed(1)
np.random.seed(1)
de.config.set_seed(1)
set_seed(1)


if __name__ == '__main__':
    t1_ = time.time()
    cfg = args_opt
    print(cfg)
    target = args_opt.device_target
    if target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target}, '
            f'Expected one of: "CPU", "GPU"'
        )

    session = onnxruntime.InferenceSession(args_opt.onnx_path, providers=providers)
    predict_data = create_eval_dataset(
        cfg.video_path, cfg.annotation_path, cfg)
    size = predict_data.get_dataset_size()
    total_target_path = get_target_path(cfg.annotation_path)
    with total_target_path.open('r') as f:
        total_target_data = json.load(f)
    results = {'results': defaultdict(list)}
    count = 0
    for data in predict_data.create_dict_iterator(output_numpy=True):
        t1 = time.time()
        x, label = data['data'][0], data['label'].tolist()
        video_ids, segments = zip(
            *total_target_data['targets'][str(label[0])])
        x_list = np.split(x, x.shape[0], axis=0)
        outputs = []
        for x in x_list:
            inputs = {session.get_inputs()[0].name: x}
            output = session.run(None, inputs)[0]
            outputs.append(output)
        outputs = np.concatenate(outputs, axis=0)

        _, locs = topk_(outputs, K=1)
        locs = locs.reshape(1, -1)

        t2 = time.time()
        print("[{} / {}] Net time: {} ms".format(count, size, (t2 - t1) * 1000))
        for j in range(0, outputs.shape[0]):
            results['results'][video_ids[j]].append({
                'segment': segments[j],
                'output': outputs[j]
            })
        count += 1

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
    # init context
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
    print(sum(correct_top1))
    print(n_ground_truth_top_1)
    print(sum(correct_top5))
    print(n_ground_truth_top_5)
    accuracy_top1 = float(sum(correct_top1)) / float(n_ground_truth_top_1)
    accuracy_top5 = float(sum(correct_top5)) / float(n_ground_truth_top_5)
    print('==================Accuracy=================\n'
          ' clip-acc : {} \ttop-1 : {} \ttop-5: {}'.format(clip_acc, accuracy_top1, accuracy_top5))
    t2_ = time.time()
    print("Total time : {} s".format(t2_ - t1_))
