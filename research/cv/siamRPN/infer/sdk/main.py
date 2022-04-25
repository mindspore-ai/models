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
"""main"""

import argparse
import os
import json
import cv2
import numpy as np
from api.infer import SdkApi
from config.config import config
from tqdm import tqdm
from shapely.geometry import Polygon


def parser_args():
    """parser_args"""
    parser = argparse.ArgumentParser(description="siamRPN inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=False,
                        default="../data/input/vot2015",
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="../data/config/siamRPN.pipeline",
        help="image file path. The default is '../data/config/siamRPN.pipeline'. ")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="./result",
        help="cache dir of inference result. The default is './result'."
    )
    arg = parser.parse_args()
    return arg


def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    cx, cy, w, h = bbox  # float type
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
    s_x = s_z * size_x / size_z
    instance_img, scale_x = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    w_x = w * scale_x
    h_x = h * scale_x
    return instance_img, w_x, h_x, scale_x


def get_exemplar_image(img, bbox, size_z, context_amount, img_mean=None):
    cx, cy, w, h = bbox
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    exemplar_img, _ = crop_and_pad(img, cx, cy, size_z, s_z, img_mean)
    return exemplar_img, scale_z, s_z


def round_up(value):
    return round(value + 1e-6 + 1000) - 1000


def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    """change img size

    :param img:rgb
    :param cx: center x
    :param cy: center y
    :param model_sz: changed size
    :param original_sz: origin size
    :param img_mean: mean of img
    :return: changed img ,scale for origin to changed
    """
    im_h, im_w, _ = img.shape
    xmin = cx - (original_sz - 1) / 2
    xmax = xmin + original_sz - 1
    ymin = cy - (original_sz - 1) / 2
    ymax = ymin + original_sz - 1
    left = int(round_up(max(0., -xmin)))
    top = int(round_up(max(0., -ymin)))
    right = int(round_up(max(0., xmax - im_w + 1)))
    bottom = int(round_up(max(0., ymax - im_h + 1)))

    xmin = int(round_up(xmin + left))
    xmax = int(round_up(xmax + left))
    ymin = int(round_up(ymin + top))
    ymax = int(round_up(ymax + top))
    r, c, k = img.shape
    if any([top, bottom, left, right]):
        # 0 is better than 1 initialization
        te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)
        te_im[top:top + r, left:left + c, :] = img
        if top:
            te_im[0:top, left:left + c, :] = img_mean
        if bottom:
            te_im[r + top:, left:left + c, :] = img_mean
        if left:
            te_im[:, 0:left, :] = img_mean
        if right:
            te_im[:, c + left:, :] = img_mean
        im_patch_original = te_im[int(ymin):int(
            ymax + 1), int(xmin):int(xmax + 1), :]
    else:
        im_patch_original = img[int(ymin):int(
            ymax + 1), int(xmin):int(xmax + 1), :]
    if not np.array_equal(model_sz, original_sz):
        # zzp: use cv to get a better speed
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    scale = model_sz / im_patch_original.shape[0]
    return im_patch, scale


def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    """ anchor generator function"""
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
        np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def box_transform_inv(anchors, offset):
    """invert transform box

    :param anchors: object
    :param offset: object
    :return: object
    """
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    offset_x, offset_y, offset_w, offset_h = offset[:,
                                                    :1], offset[:, 1:2], offset[:, 2:3], offset[:, 3:],

    box_cx = anchor_w * offset_x + anchor_xctr
    box_cy = anchor_h * offset_y + anchor_yctr
    box_w = anchor_w * np.exp(offset_w)
    box_h = anchor_h * np.exp(offset_h)
    box = np.hstack([box_cx, box_cy, box_w, box_h])
    return box


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


def softmax(y):
    """softmax of numpy"""
    x = y.copy()
    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


def judge_failures(pred_bbox, gt_bbox, threshold=0):
    """" judge whether to fail or not """
    if len(gt_bbox) == 4:
        if iou(np.array(pred_bbox).reshape(-1, 4), np.array(gt_bbox).reshape(-1, 4)) > threshold:
            return False
    else:
        poly_pred = Polygon(np.array([[pred_bbox[0], pred_bbox[1]],
                                      [pred_bbox[2], pred_bbox[1]],
                                      [pred_bbox[2], pred_bbox[3]],
                                      [pred_bbox[0], pred_bbox[3]]
                                      ])).convex_hull
        poly_gt = Polygon(np.array(gt_bbox).reshape(4, 2)).convex_hull
        inter_area = poly_gt.intersection(poly_pred).area
        overlap = inter_area / (poly_gt.area + poly_pred.area - inter_area)
        if overlap > threshold:
            return False
    return True


def calculate_accuracy_failures(pred_trajectory, gt_trajectory,
                                bound=None):
    '''
    args:
    pred_trajectory:list of bbox
    gt_trajectory: list of bbox ,shape == pred_trajectory
    bound :w and h of img
    return :
    overlaps:list ,iou value in pred_trajectory
    acc : mean iou value
    failures: failures point in pred_trajectory
    num_failures: number of failres
    '''

    overlaps = []
    failures = []

    for i, pred_traj in enumerate(pred_trajectory):
        if len(pred_traj) == 1:

            if pred_trajectory[i][0] == 2:
                failures.append(i)
            overlaps.append(float("nan"))

        else:
            if bound is not None:
                poly_img = Polygon(np.array([[0, 0],
                                             [0, bound[1]],
                                             [bound[0], bound[1]],
                                             [bound[0], 0]])).convex_hull

            if len(gt_trajectory[i]) == 8:

                poly_pred = Polygon(np.array([[pred_trajectory[i][0], pred_trajectory[i][1]],
                                              [pred_trajectory[i][2], pred_trajectory[i][1]],
                                              [pred_trajectory[i][2], pred_trajectory[i][3]],
                                              [pred_trajectory[i][0], pred_trajectory[i][3]]
                                              ])).convex_hull
                poly_gt = Polygon(
                    np.array(gt_trajectory[i]).reshape(4, 2)).convex_hull
                if bound is not None:
                    gt_inter_img = poly_gt.intersection(poly_img)
                    pred_inter_img = poly_pred.intersection(poly_img)
                    inter_area = gt_inter_img.intersection(pred_inter_img).area
                    overlap = inter_area / \
                        (gt_inter_img.area + pred_inter_img.area - inter_area)
                else:
                    inter_area = poly_gt.intersection(poly_pred).area
                    overlap = inter_area / \
                        (poly_gt.area + poly_pred.area - inter_area)
            elif len(gt_trajectory[i]) == 4:

                overlap = iou(np.array(pred_trajectory[i]).reshape(-1, 4), np.array(gt_trajectory[i]).reshape(-1, 4))
            overlaps.append(overlap)
    acc = 0
    num_failures = len(failures)
    if overlaps:
        acc = np.nanmean(overlaps)
    return acc, overlaps, failures, num_failures


def calculate_expected_overlap(fragments, fweights):
    """ compute expected iou """
    max_len = fragments.shape[1]
    expected_overlaps = np.zeros((max_len), np.float32)
    expected_overlaps[0] = 1

    # TODO Speed Up
    for i in range(1, max_len):
        mask = np.logical_not(np.isnan(fragments[:, i]))
        if np.any(mask):
            fragment = fragments[mask, 1:i+1]
            seq_mean = np.sum(fragment, 1) / fragment.shape[1]
            expected_overlaps[i] = np.sum(seq_mean *
                                          fweights[mask]) / np.sum(fweights[mask])
    return expected_overlaps


def iou(box1, box2):
    """ compute iou """
    box1, box2 = box1.copy(), box2.copy()
    N = box1.shape[0]
    K = box2.shape[0]
    box1 = np.array(box1.reshape((N, 1, 4))) + \
        np.zeros((1, K, 4))  # box1=[N,K,4]
    box2 = np.array(box2.reshape((1, K, 4))) + \
        np.zeros((N, 1, 4))  # box1=[N,K,4]
    x_max = np.max(np.stack((box1[:, :, 0], box2[:, :, 0]), axis=-1), axis=2)
    x_min = np.min(np.stack((box1[:, :, 2], box2[:, :, 2]), axis=-1), axis=2)
    y_max = np.max(np.stack((box1[:, :, 1], box2[:, :, 1]), axis=-1), axis=2)
    y_min = np.min(np.stack((box1[:, :, 3], box2[:, :, 3]), axis=-1), axis=2)
    tb = x_min-x_max
    lr = y_min-y_max
    tb[np.where(tb < 0)] = 0
    lr[np.where(lr < 0)] = 0
    over_square = tb*lr
    all_square = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1]) + (box2[:, :, 2] - \
                                            box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1]) - over_square
    return over_square / all_square


def calculate_eao(dataset_name, all_failures, all_overlaps, gt_traj_length, skipping=5):
    '''
    input:dataset name
    all_failures: type is list , index of failure
    all_overlaps: type is  list , length of list is the length of all_failures
    gt_traj_length: type is list , length of list is the length of all_failures
    skipping：number of skipping per failing
    '''
    if dataset_name == "VOT2016":

        low = 108
        high = 371

    elif dataset_name == "VOT2015":
        low = 108
        high = 371

    fragment_num = sum([len(x)+1 for x in all_failures])
    max_len = max([len(x) for x in all_overlaps])
    tags = [1] * max_len
    seq_weight = 1 / (1 + 1e-10)  # division by zero

    eao = {}

    # prepare segments
    fweights = np.ones(fragment_num, dtype=np.float32) * np.nan
    fragments = np.ones((fragment_num, max_len), dtype=np.float32) * np.nan
    seg_counter = 0
    for traj_len, failures, overlaps in zip(gt_traj_length, all_failures, all_overlaps):
        if failures:
            points = [x+skipping for x in failures if
                      x+skipping <= len(overlaps)]
            points.insert(0, 0)
            for i, _ in enumerate(points):
                if i != len(points) - 1:
                    fragment = np.array(
                        overlaps[points[i]:points[i+1]+1], dtype=np.float32)
                    fragments[seg_counter, :] = 0
                else:
                    fragment = np.array(overlaps[points[i]:], dtype=np.float32)
                fragment[np.isnan(fragment)] = 0
                fragments[seg_counter, :len(fragment)] = fragment
                if i != len(points) - 1:
                    tag_value = tags[points[i]:points[i+1]+1]
                    w = sum(tag_value) / (points[i+1] - points[i]+1)
                    fweights[seg_counter] = seq_weight * w
                else:
                    tag_value = tags[points[i]:len(overlaps)]
                    w = sum(tag_value) / (traj_len - points[i]+1e-16)
                    fweights[seg_counter] = seq_weight * w
                seg_counter += 1
        else:
            # no failure
            max_idx = min(len(overlaps), max_len)
            fragments[seg_counter, :max_idx] = overlaps[:max_idx]
            tag_value = tags[0: max_idx]
            w = sum(tag_value) / max_idx
            fweights[seg_counter] = seq_weight * w
            seg_counter += 1

    expected_overlaps = calculate_expected_overlap(fragments, fweights)
    print(len(expected_overlaps))
    # calculate eao
    weight = np.zeros((len(expected_overlaps)))
    weight[low-1:high-1+1] = 1
    expected_overlaps = np.array(expected_overlaps, dtype=np.float32)
    is_valid = np.logical_not(np.isnan(expected_overlaps))
    eao_ = np.sum(expected_overlaps[is_valid] *
                  weight[is_valid]) / np.sum(weight[is_valid])
    eao = eao_
    return eao

class SiamRPNTracker:
    """ Tracker for SiamRPN"""
    def __init__(self):
        valid_scope = 2 * config.valid_scope + 1
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        valid_scope)

        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(
            np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.shape = frame.shape
        self.pos = np.array(
            [bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2])  # center x, center y, zero based

        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
        self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2,
                              bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])

        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        # get exemplar img
        self.img_mean = np.mean(frame, axis=(0, 1))

        exemplar_img, _, _ = get_exemplar_image(frame, self.bbox,
                                                config.exemplar_size, config.context_amount, self.img_mean)
        exemplar_img = exemplar_img.transpose((2, 0, 1)).astype(np.float32)
        exemplar_img = np.expand_dims(exemplar_img, axis=0)
        return exemplar_img

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image
        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        self.img_mean = np.mean(frame, axis=(0, 1))
        instance_img_np, _, _, scale_x = get_instance_image(frame, self.bbox, config.exemplar_size,
                                                            config.instance_size,
                                                            config.context_amount, self.img_mean)
        self.scale_x = scale_x
        instance_img_np = instance_img_np.transpose(
            (2, 0, 1)).astype(np.float32)
        instance_img_np = np.expand_dims(instance_img_np, axis=0)
        return instance_img_np

    def postprocess(self, pred_score, pred_regression):
        """postprocess of prediction"""
        pred_score = np.frombuffer(pred_score, dtype=np.float32)
        pred_regression = np.frombuffer(pred_regression, dtype=np.float32)
        pred_conf = pred_score.reshape(
            (config.anchor_num * config.score_size * config.score_size, 2))
        pred_offset = pred_regression.reshape(
            (config.anchor_num * config.score_size * config.score_size, 4))
        delta = pred_offset
        box_pred = box_transform_inv(self.anchors, delta)
        score_pred = softmax(pred_conf)[:, 1]

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

        s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) /
                     (sz_wh(self.target_sz * self.scale_x)))  # scale penalty
        r_c = change((self.target_sz[0] / self.target_sz[1]) /
                     (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
        pscore = penalty * score_pred
        pscore = pscore * (1 - config.window_influence) + \
            self.window * config.window_influence
        best_pscore_id = np.argmax(pscore)

        target = box_pred[best_pscore_id, :] / self.scale_x

        lr = penalty[best_pscore_id] * \
            score_pred[best_pscore_id] * config.lr_box

        res_x = np.clip(target[0] + self.pos[0], 0, self.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, self.shape[0])

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])
        self.bbox = (
            np.clip(bbox[0], 0, self.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, self.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, self.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, self.shape[0]).astype(np.float64))
        return self.bbox, score_pred[best_pscore_id]


def write_result(path, data):
    f = open(path, "w")
    for box in data:
        f.write(str(box)+'\n')
    f.close()


def image_inference(pipeline_path, dataset, result_dir):
    """image_inference"""
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print("\nBegin to inference for {}.\n".format(dataset))
    direct_file = os.path.join(dataset, 'list.txt')
    with open(direct_file, 'r') as f:
        direct_lines = f.readlines()
    video_names = np.sort([x.split('\n')[0] for x in direct_lines])
    video_paths = [os.path.join(dataset, x) for x in video_names]
    tracker = SiamRPNTracker()
    # ------------ starting validation  -----------
    results = {}
    accuracy = 0
    all_overlaps = []
    all_failures = []
    gt_lenth = []
    for video_name in tqdm(video_names, total=len(video_paths)):
        # ------------ prepare groundtruth  -----------
        groundtruth_path = os.path.join(dataset, video_name, 'groundtruth.txt')
        with open(groundtruth_path, 'r') as f:
            boxes = f.readlines()
        if ',' in boxes[0]:
            boxes = [list(map(float, box.split(','))) for box in boxes]
        else:
            boxes = [list(map(int, box.split())) for box in boxes]

        gt = boxes.copy()
        frames = [os.path.join(dataset, video_name, 'color', x) for x in np.sort(
            os.listdir(os.path.join(dataset, video_name, 'color')))]
        frames = [x for x in frames if '.jpg' in x]
        template_idx = 0
        res = []
        if not os.path.exists(os.path.join(result_dir, video_name)):
            os.makedirs(os.path.join(result_dir, video_name))
        result_path = os.path.join(result_dir, video_name, "prediction.txt")
        template_plugin_id = 0
        detection_plugin_id = 1
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            frame = cv2.imdecode(np.fromfile(
                frame, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            h, w = frame.shape[0], frame.shape[1]
            print('processing {}/{}'.format(idx + 1, len(frames)))
            if idx == template_idx:
                box = get_axis_aligned_bbox(boxes[idx])
                template = tracker.init(frame, box)
                res.append([1])
            elif idx < template_idx:
                res.append([0])
            else:
                detection = tracker.update(frame)
                sdk_api.send_tensor_input(config.sdk_pipeline_name, template_plugin_id,
                                          "appsrc0", template.tobytes(), [
                                              1, 3, 127, 127],
                                          0)
                sdk_api.send_img_input(config.sdk_pipeline_name,
                                       detection_plugin_id, "appsrc1",
                                       detection.tobytes(), detection.shape)
                result = sdk_api.get_result(config.sdk_pipeline_name)
                cout, rout = result[0], result[1]
                bbox, _ = tracker.postprocess(cout, rout)
                bbox = np.array(bbox)
                bbox = list((bbox[0] - bbox[2] / 2 + 1 / 2, bbox[1] - bbox[3] / 2 + 1 / 2, \
                             bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2))
                if judge_failures(bbox, boxes[idx], 0):
                    res.append([2])
                    print('fail')
                    template_idx = min(idx + 5, len(frames) - 1)
                else:
                    res.append(bbox)
        write_result(result_path, res)
        acc, overlaps, failures, num_failures = calculate_accuracy_failures(res, gt, [w, h])
        accuracy += acc
        result1 = {}
        result1['acc'] = acc
        result1['num_failures'] = num_failures
        results[video_name] = result1
        all_overlaps.append(overlaps)
        all_failures.append(failures)
        gt_lenth.append(len(frames))
        print(acc, overlaps, num_failures)
    all_length = sum([len(x) for x in all_overlaps])
    robustness = sum([len(x) for x in all_failures]) / all_length * 100
    eao = calculate_eao("VOT2016", all_failures, all_overlaps, gt_lenth)
    result1 = {}
    result1['accuracy'] = accuracy / float(len(video_paths))
    result1['robustness'] = robustness
    result1['eao'] = eao
    results['all_videos'] = result1
    print('accuracy is ', accuracy / float(len(video_paths)))
    print('robustness is ', robustness)
    print('eao is ', eao)
    json.dump(results, open('./result_val.json', 'w'))


if __name__ == "__main__":
    args = parser_args()
    image_inference(args.pipeline_path, args.img_path, args.infer_result_dir)
