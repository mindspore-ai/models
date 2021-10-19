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
''' eval PoseEstNet '''
import csv
import math
import os
import argparse
import numpy as np

from utils.inference import SdkInfer, get_posenet_preds
from utils.transforms import image_proc, flip_back, flip_pairs


def calc_dists(preds, target, normalize):
    ''' calc dists '''
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    return -1


def accuracy(output, target, hm_type='gaussian'):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_posenet_preds(output, trans_back=False)
        target, _ = get_posenet_preds(target, trans_back=False)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def eval_posenet(label_path, imgs_path, pipline_path, FLIP_TEST=True, SHIFT_HEATMAP=True):
    ''' start eval '''
    stream = SdkInfer(pipline_path)
    stream.init_stream()

    label_csv = open(label_path)
    reader = csv.reader(label_csv, delimiter=',')
    hash_annot = {}
    data = []
    for row in reader:
        img_name = row[0]
        width = int(row[1])
        height = int(row[2])
        joints = []
        for j in range(36):
            joint = [int(row[j * 3 + 3]), int(row[j * 3 + 4]),
                     int(row[j * 3 + 5])]
            joints.append(joint)
        hash_annot[img_name] = (width, height, joints)
    center_joints = []
    scale_joints = []
    all_preds = np.zeros(
        (len(hash_annot), 36, 3),
        dtype=np.float32
    )
    batch_count = 0
    idx = 0
    for k in sorted(hash_annot.keys()):
        image_name = k
        if not image_name.lower().endswith((".jpg", "jpeg")):
            continue
        img_path = os.path.join(imgs_path, image_name)
        _, pe_input, center, scale = image_proc(img_path)
        pn_id = stream.send_package_buf(b'PoseEstNet0', np.expand_dims(
            pe_input.astype(np.float32), axis=0), 0)
        infer_result = stream.get_result(b'PoseEstNet0', pn_id)

        if FLIP_TEST:
            input_flipped = np.flip(np.expand_dims(
                pe_input.astype(np.float32), axis=0), 3)
            pn_flipped_id = stream.send_package_buf(
                b'PoseEstNet0', input_flipped, 0)
            outputs_flipped = stream.get_result(b'PoseEstNet0', pn_flipped_id)
            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped
            output_flipped = flip_back(np.array(output_flipped), flip_pairs)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if SHIFT_HEATMAP:  # true
                output_flipped_copy = output_flipped
                output_flipped[:, :, :,
                               1:] = output_flipped_copy[:, :, :, 0:-1]
            infer_result = (infer_result + output_flipped) * 0.5
        data.append(infer_result)
        center_joints.append(center)
        scale_joints.append(scale)
        if batch_count == 31:
            output = np.array(data, np.float32)
            output = np.stack(output, axis=1).squeeze(axis=0)
            preds = get_posenet_preds(
                output, center=center_joints, scale=scale_joints)

            all_preds[idx:idx + batch_count + 1, :, 0:3] = preds[:, :, 0:3]
            print(f'-------- Test: [{int((idx+1)/32 + 1)}/{int(len(hash_annot)/32)}] ---------')
            name_values = _evaluate(all_preds, label_path)
            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, 'PoseEstNet')
            else:
                _print_name_value(name_values, 'PoseEstNet')
            data = []
            center_joints = []
            scale_joints = []
            idx += batch_count + 1
            batch_count = 0
        else:
            batch_count += 1
    stream.destroy()


def _print_name_value(name_value, full_arch_name):
    ''' print accuracy '''
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    print(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    print('| --- ' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    print(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def _evaluate(preds, label_path):
    ''' get accuracy set'''
    SC_BIAS = 0.25
    threshold = 0.5

    gts = []
    viss = []
    area_sqrts = []
    with open(label_path) as annot_file:
        reader = csv.reader(annot_file, delimiter=',')
        for row in reader:
            joints = []
            vis = []
            top_lft = btm_rgt = [int(row[3]), int(row[4])]
            for j in range(36):
                joint = [int(row[j * 3 + 3]), int(row[j * 3 + 4]),
                         int(row[j * 3 + 5])]
                joints.append(joint)
                vis.append(joint[2])
                if joint[0] < top_lft[0]:
                    top_lft[0] = joint[0]
                if joint[1] < top_lft[1]:
                    top_lft[1] = joint[1]
                if joint[0] > btm_rgt[0]:
                    btm_rgt[0] = joint[0]
                if joint[1] > btm_rgt[1]:
                    btm_rgt[1] = joint[1]
            gts.append(joints)
            viss.append(vis)
            area_sqrts.append(
                math.sqrt((btm_rgt[0] - top_lft[0] + 1) * (btm_rgt[1] - top_lft[1] + 1)))

    jnt_visible = np.array(viss, dtype=np.int)
    jnt_visible = np.transpose(jnt_visible)
    pos_pred_src = np.transpose(preds, [1, 2, 0])
    pos_gt_src = np.transpose(gts, [1, 2, 0])
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    area_sqrts = np.linalg.norm(area_sqrts, axis=0)
    area_sqrts *= SC_BIAS
    scale = np.multiply(area_sqrts, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                      jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    # save
    rng = np.arange(0, 0.5 + 0.01, 0.01)
    pckAll = np.zeros((len(rng), 36))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
                                 jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

    name_value = {
        'Wheel': (1.0 / 4.0) * (PCKh[0] + PCKh[1] + PCKh[18] + PCKh[19]),
        'Fender': (1.0 / 16.0) * (PCKh[2] + PCKh[3] + PCKh[4] + PCKh[5] + PCKh[6] + PCKh[7] + PCKh[8] +
                                  PCKh[9] + PCKh[20] + PCKh[21] + PCKh[22] + PCKh[23] + PCKh[24] +
                                  PCKh[25] + PCKh[26] + PCKh[27]),
        'Back': (1.0 / 4.0) * (PCKh[10] + PCKh[11] + PCKh[28] + PCKh[29]),
        'Front': (1.0 / 4.0) * (PCKh[16] + PCKh[17] + PCKh[34] + PCKh[35]),
        'WindshieldBack': (1.0 / 4.0) * (PCKh[12] + PCKh[13] + PCKh[30] + PCKh[31]),
        'WindshieldFront': (1.0 / 4.0) * (PCKh[14] + PCKh[15] + PCKh[32] + PCKh[33]),
        'Mean': np.sum(PCKh * jnt_ratio),
        'Mean@0.1': np.sum(pckAll[11, :] * jnt_ratio)
    }

    return name_value


def generate_target(joints, joints_vis, target_type='gaussian'):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    target_weight = np.ones((36, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((36, 64, 64), dtype=np.float32)

    assert target_type == 'gaussian', \
        'Only support gaussian map now!'

    if target_type == 'gaussian':
        # tmp_size = sigma * 3
        tmp_size = 6
        for joint_id in range(36):
            feat_stride = [4, 4]  # 256,256 / 64,64
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= 64 or ul[1] >= 64 \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * 2 ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], 64) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], 64) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], 64)
            img_y = max(0, ul[1]), min(br[1], 64)

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]
                                 :img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return target, target_weight

parser = argparse.ArgumentParser(description='Eval PoseEstNet')
parser.add_argument('--img_path', type=str, default='../data/PoseEstNet/veri/images/image_test')
parser.add_argument('--label_path', type=str, default='../data/PoseEstNet/veri/annot/label_test.csv')
parser.add_argument('--pipline_path', type=str, default='../pipline/pamtri.pipline')
args = parser.parse_args()
if __name__ == '__main__':
    eval_posenet(args.label_path, args.img_path, args.pipline_path)
