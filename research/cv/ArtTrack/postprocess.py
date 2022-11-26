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
import argparse

import numpy as np
from numpy import array as arr

from config import check_config
from src.dataset.mpii import MPII
from src.model.predict import extract_cnn_output, argmax_pose_predict

def parse_args():
    '''
    parse_args
    '''
    parser = argparse.ArgumentParser(description='get_acc')
    parser.add_argument('--result_path', required=False, default="./result_Files/", help='Location of result.')
    parser.add_argument('--ratios_path', required=False, default="./mpii/ratios.npy", help='Location of ratios.npy.')
    parser.add_argument('--cfg', required=False, default="./config/mpii_eval_ascend.yaml",
                        help='Location of mpii_test.yaml.')
    opt_args = parser.parse_args()
    return opt_args

def test(args, cfg=None):
    """
    entry for predicting single mpii
    Args:
        cfg: config
    """

    ratios = np.load(args.ratios_path)
    num_images = ratios.shape[0]
    predictions = np.zeros((num_images,), dtype=np.object)
    result_path = args.result_path
    out_shape = [1, 14, 52, 40]
    pairwise_pred_shape = [1, 28, 52, 40]
    for i in range(num_images):
        f1 = os.path.join(result_path, str(i)+"_0.bin")
        f2 = os.path.join(result_path, str(i)+"_1.bin")
        out = np.fromfile(f1, np.float32).reshape(out_shape).transpose([0, 2, 3, 1])
        locref = None
        pairwise_pred = np.fromfile(f2, np.float32).reshape(pairwise_pred_shape).transpose([0, 2, 3, 1])
        scmap, locref, _ = extract_cnn_output(out, locref, pairwise_pred, cfg)
        pose = argmax_pose_predict(scmap, locref, cfg.stride)

        pose_refscale = np.copy(pose)
        pose_refscale[:, 0:2] /= cfg.global_scale
        ratio = ratios[i]
        pose_refscale[:, 0] /= ratio[0]
        pose_refscale[:, 1] /= ratio[1]
        predictions[i] = pose_refscale

    return predictions

def enclosing_rect(points):
    """
    enclose rectangle
    """
    xs = points[:, 0]
    ys = points[:, 1]
    return np.array([np.amin(xs), np.amin(ys), np.amax(xs), np.amax(ys)])


def rect_size(rect):
    """
    get rectangle size
    """
    return np.array([rect[2] - rect[0], rect[3] - rect[1]])


def print_results(pck, cfg):
    """
    print result
    """
    s = ""
    for heading in cfg.all_joints_names + ["total"]:
        s += " & " + heading
    print(s)

    s = ""
    all_joint_ids = cfg.all_joints + [np.arange(cfg.num_joints)]
    for j_ids in all_joint_ids:
        j_ids_np = arr(j_ids)
        pck_av = np.mean(pck[j_ids_np])
        s += " & {0:.1f}".format(pck_av)
    print(s)

def eval_pck(pred, cfg=None):
    """
    eval mpii entry
    """
    dataset = MPII(cfg)

    joints = np.array([pred])
    pck_ratio_thresh = cfg.pck_threshold

    num_joints = cfg.num_joints
    num_images = joints.shape[1]

    pred_joints = np.zeros((num_images, num_joints, 2))
    gt_joints = np.zeros((num_images, num_joints, 2))
    pck_thresh = np.zeros((num_images, 1))
    gt_present_joints = np.zeros((num_images, num_joints))

    for k in range(num_images):
        pred = joints[0, k]
        gt = dataset.data[k].joints[0]
        if gt.shape[0] == 0:
            continue
        gt_joint_ids = gt[:, 0].astype('int32')
        rect = enclosing_rect(gt[:, 1:3])
        pck_thresh[k] = pck_ratio_thresh * np.amax(rect_size(rect))

        gt_present_joints[k, gt_joint_ids] = 1
        gt_joints[k, gt_joint_ids, :] = gt[:, 1:3]
        pred_joints[k, :, :] = pred[:, 0:2]

    dists = np.sqrt(np.sum((pred_joints - gt_joints) ** 2, axis=2))
    correct = dists <= pck_thresh

    num_all = np.sum(gt_present_joints, axis=0)

    num_correct = np.zeros((num_joints,))
    for j_id in range(num_joints):
        num_correct[j_id] = np.sum(correct[gt_present_joints[:, j_id] == 1, j_id], axis=0)

    pck = num_correct / num_all * 100.0

    print_results(pck, cfg)

def main(args):
    cfg = check_config(args.cfg, None)
    predictions = test(args, cfg)
    eval_pck(predictions, cfg)

if __name__ == '__main__':
    main(parse_args())
