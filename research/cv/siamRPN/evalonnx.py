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
"""eval vot"""

import argparse
import os
import json
import sys
import time
import numpy as np
import onnxruntime as ort
import mindspore as ms
from mindspore import context, ops
from mindspore import Tensor
from src import evaluation as eval_
from src.config import config
from src.util import get_exemplar_image, get_instance_image, box_transform_inv
from src.generate_anchors import generate_anchors
from tqdm import tqdm
import cv2

sys.path.append(os.getcwd())


def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    return session


def change(r):
    return np.maximum(r, 1. / r)


def sz(w, h):
    pad = (w + h) * 0.5
    sz2 = (w + pad) * (h + pad)
    return np.sqrt(sz2)


def sz_wh(wh):
    pad = (wh[0] + wh[1]) * 0.5
    sz2 = (wh[0] + pad) * (wh[1] + pad)
    return np.sqrt(sz2)


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = len(region)
    region = np.array(region)
    if nv == 8:
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
             np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        x = x1
        y = y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]

    return x, y, w, h


def reshapeimg1(img):
    img = Tensor(img, ms.float32)
    img = img.transpose((2, 0, 1))
    img = img.asnumpy()
    img = img.reshape(1, 3, 127, 127)
    return img


def reshapeimg2(img):
    img = Tensor(img, ms.float32)
    img = img.transpose((2, 0, 1))
    img = img.asnumpy()
    img = img.reshape(1, 3, 255, 255)
    return img


def calculate(bbox):
    gbox = np.array(bbox)
    gbox = list((gbox[0] - gbox[2] / 2 + 1 / 2, gbox[1] - gbox[3] / 2 + 1 / 2,
                 gbox[0] + gbox[2] / 2 - 1 / 2, gbox[1] + gbox[3] / 2 - 1 / 2))
    return gbox


def show(accuracy, video_paths, robustness, eao):
    print('accuracy is ', accuracy / float(len(video_paths)))
    print('robustness is ', robustness)
    print('eao is ', eao)


def predscore(pred_score):
    pred_score = Tensor(pred_score)
    softmax = ops.Softmax(axis=2)
    pred_score = softmax(pred_score)[0, :, 1]
    pred_score = pred_score.asnumpy()
    return pred_score


def resshow(target, pos, frame, origin_target_sz, lr, target_sz):
    res_x = np.clip(target[0] + pos[0], 0, frame.shape[1])
    res_y = np.clip(target[1] + pos[1], 0, frame.shape[0])
    res_w = np.clip(target_sz[0] * (1 - lr) + target[2] * lr,
                    config.min_scale * origin_target_sz[0],
                    config.max_scale * origin_target_sz[0])
    res_h = np.clip(target_sz[1] * (1 - lr) + target[3] * lr,
                    config.min_scale * origin_target_sz[1],
                    config.max_scale * origin_target_sz[1])
    return res_x, res_y, res_w, res_h


def bboxshow(bbox, frame):
    bbox = (
        np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
        np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
        np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
        np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))
    return bbox


def result1show(acc, num_failures, frames, duration):
    result1 = {}
    result1['acc'] = acc
    result1['num_failures'] = num_failures
    result1['fps'] = round(len(frames) / duration, 3)
    return result1


def test(model_path, data_path, save_name):
    session = create_session(model_path, "GPU")
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
    direct_file = os.path.join(data_path, 'list.txt')
    with open(direct_file, 'r') as f:
        direct_lines = f.readlines()
    video_names = np.sort([x.split('\n')[0] for x in direct_lines])
    video_paths = [os.path.join(data_path, x) for x in video_names]
    results = {}
    accuracy = 0
    all_overlaps = []
    all_failures = []
    gt_lenth = []
    for video_path in tqdm(video_paths, total=len(video_paths)):
        groundtruth_path = os.path.join(video_path, 'groundtruth.txt')
        with open(groundtruth_path, 'r') as f:
            boxes = f.readlines()
        if ',' in boxes[0]:
            boxes = [list(map(float, box.split(','))) for box in boxes]
        else:
            boxes = [list(map(int, box.split())) for box in boxes]
        gt = boxes.copy()
        gt[:][2] = gt[:][0] + gt[:][2]
        gt[:][3] = gt[:][1] + gt[:][3]
        frames = [os.path.join(video_path, 'color', x) for x in np.sort(os.listdir(os.path.join(video_path, 'color')))]
        frames = [x for x in frames if '.jpg' in x]
        tic = time.perf_counter()
        template_idx = 0
        valid_scope = 2 * config.valid_scope + 1
        anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                   config.anchor_ratios,
                                   valid_scope)
        window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                         [config.anchor_num, 1, 1]).flatten()
        res = []
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            frame = cv2.imdecode(np.fromfile(frame, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            h, w = frame.shape[0], frame.shape[1]
            if idx == template_idx:
                bbox = get_axis_aligned_bbox(boxes[idx])
                pos = np.array(
                    [bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2])  # center x, center y, zero based
                target_sz = np.array([bbox[2], bbox[3]])
                bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])
                origin_target_sz = np.array([bbox[2], bbox[3]])
                img_mean = np.mean(frame, axis=(0, 1))
                exemplar_img, _, _ = get_exemplar_image(frame, bbox,
                                                        config.exemplar_size, config.context_amount, img_mean)
                exemplar_img = reshapeimg1(exemplar_img)
                res.append([1])
            elif idx < template_idx:
                res.append([0])
            else:
                instance_img_np, _, _, scale_x = get_instance_image(frame, bbox, config.exemplar_size,
                                                                    config.instance_size,
                                                                    config.context_amount, img_mean)
                instance_img_np = reshapeimg2(instance_img_np)
                pred_score, pred_regress = session.run(outname, {inname[0]: exemplar_img, inname[1]: instance_img_np})
                pred_score = predscore(pred_score)
                delta = pred_regress[0]
                box_pred = box_transform_inv(anchors, delta)
                s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) / (sz_wh(target_sz * scale_x)))  # scale penalty
                r_c = change(
                    (target_sz[0] / target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
                penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
                pscore = penalty * pred_score
                pscore = pscore * (1 - config.window_influence) + window * config.window_influence
                best_pscore_id = np.argmax(pscore)
                target = box_pred[best_pscore_id, :] / scale_x
                lr = penalty[best_pscore_id] * pred_score[best_pscore_id] * config.lr_box
                res_x, res_y, res_w, res_h = resshow(target, pos, frame, origin_target_sz, lr, target_sz)
                pos = np.array([res_x, res_y])
                target_sz = np.array([res_w, res_h])
                bbox = np.array([res_x, res_y, res_w, res_h])
                bbox = bboxshow(bbox, frame)
                gbox = calculate(bbox)
                if eval_.judge_failures(gbox, boxes[idx], 0):
                    res.append([2])
                    template_idx = min(idx + 5, len(frames) - 1)
                else:
                    res.append(gbox)
        duration = time.perf_counter() - tic
        acc, overlaps, failures, num_failures = eval_.calculate_accuracy_failures(res, gt, [w, h])
        accuracy += acc
        result1 = result1show(acc, num_failures, frames, duration)
        results[video_path.split('/')[-1]] = result1
        all_overlaps.append(overlaps)
        all_failures.append(failures)
        gt_lenth.append(len(frames))
    all_length = sum([len(x) for x in all_overlaps])
    robustness = sum([len(x) for x in all_failures]) / all_length * 100
    eao = eval_.calculate_eao("VOT2015", all_failures, all_overlaps, gt_lenth)
    result1 = {}
    result1['accuracy'] = accuracy / float(len(video_paths))
    result1['robustness'] = robustness
    result1['eao'] = eao
    results['all_videos'] = result1
    show(accuracy, video_paths, robustness, eao)
    json.dump(results, open(save_name, 'w'))


def parse_args():
    '''parse_args'''
    parser = argparse.ArgumentParser(description='Mindspore SiameseRPN Infering')
    parser.add_argument('--device_target', type=str, default='GPU', choices=('Ascend', 'GPU'), help='run platform')
    parser.add_argument('--device_id', type=int, default=0, help='DEVICE_ID')
    # choose dataset_path vot2016 or vot vot2015
    parser.add_argument('--dataset_path', type=str, default='vot2015', help='Dataset path')
    parser.add_argument('--checkpoint_path', type=str, default='siamrpn.onnx', help='checkpoint of siamRPN')
    parser.add_argument('--filename', type=str, default='onnx2015', help='save result file')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == '__main__':
    args = parse_args()
    if args.device_target == 'GPU':
        device_id = args.device_id
        context.set_context(device_id=device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    model_file_path = args.checkpoint_path
    data_file_path = args.dataset_path
    save_file_name = args.filename
    test(model_path=model_file_path, data_path=data_file_path, save_name=save_file_name)
