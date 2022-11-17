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


import os
import math
import operator
from functools import reduce
import time
import numpy as np
import cv2
import onnxruntime
from mindspore import context
from src.dataset import test_dataset_creator
from src.model_utils.config import config


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                    save_graphs_path=".")


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sort_to_clockwise(points):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    clockwise_points = sorted(points, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    return clockwise_points


def write_result_as_txt(img_name, bboxes, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    filename = os.path.join(path, 'res_{}.txt'.format(os.path.splitext(img_name)[0]))
    lines = []
    for _, bbox in enumerate(bboxes):
        bbox = bbox.reshape(-1, 2)
        bbox = np.array(list(sort_to_clockwise(bbox)))[[3, 0, 1, 2]].copy().reshape(-1)
        values = [int(v) for v in bbox]
        line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
        lines.append(line)
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)

def test():
    from src.PSENET.pse import pse

    local_path = ""
    print('local_path: ', local_path)

    if not os.path.isdir('{}./res/submit_ic15/'.format(local_path)):
        os.makedirs('{}./res/submit_ic15/'.format(local_path))
    if not os.path.isdir('{}./res/vis_ic15/'.format(local_path)):
        os.makedirs('{}./res/vis_ic15/'.format(local_path))
    ds = test_dataset_creator()

    config.INFERENCE = True
    print(config.device_target)
    if config.device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif config.device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {config.device_target}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = onnxruntime.InferenceSession(config.onnx_path, provider_options=providers)
    get_data_time = AverageMeter()
    model_run_time = AverageMeter()
    post_process_time = AverageMeter()

    end_pts = time.time()
    iters = ds.create_tuple_iterator(output_numpy=True)
    count = 0
    for data in iters:
        count += 1
        # get data
        img, img_resized, img_name = data
        img = img[0].astype(np.uint8).copy()
        img_name = img_name[0]

        get_data_pts = time.time()
        get_data_time.update(get_data_pts - end_pts)

        inputs = {session.get_inputs()[0].name: img_resized}
        score, kernels = session.run(None, inputs)
        score = np.squeeze(score)
        kernels = np.squeeze(kernels)

        model_run_pts = time.time()
        model_run_time.update(model_run_pts - get_data_pts)

        # post-process
        pred = pse(kernels, 5.0)
        scale = max(img.shape[:2]) * 1.0 / config.INFER_LONG_SIZE
        label = pred
        label_num = np.max(label) + 1
        bboxes = []

        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]
            if points.shape[0] < 600:
                continue

            score_i = np.mean(score[label == i])
            if score_i < 0.93:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale
            bbox = bbox.astype('int32')
            cv2.drawContours(img, [bbox], 0, (0, 255, 0), 3)
            bboxes.append(bbox)

        post_process_pts = time.time()
        post_process_time.update(post_process_pts - model_run_pts)

        if count == 1:
            get_data_time.reset()
            model_run_time.reset()
            post_process_time.reset()

        end_pts = time.time()

        # save res
        cv2.imwrite('{}./res/vis_ic15/{}'.format(local_path, img_name), img[:, :, [2, 1, 0]].copy())
        write_result_as_txt(img_name, bboxes, '{}./res/submit_ic15/'.format(local_path))


if __name__ == "__main__":
    test()
