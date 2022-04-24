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
"""eval utils"""
import io as sysio
import math

import numpy as np
from numba import cuda
from numba import float32 as numba_float32
from numba import jit as numba_jit


@numba_jit(nopython=True)
def div_up(m, n):
    """div_up"""
    return m // n + (m % n > 0)


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def trangle_area(a, b, c):
    """triangle_area"""
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def area(int_pts, num_of_inter):
    """area"""
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                                     int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    """sort vertex in convex polygon"""
    if num_of_inter > 0:
        center = cuda.local.array((2,), dtype=numba_float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2,), dtype=numba_float32)
        vs = cuda.local.array((16,), dtype=numba_float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@cuda.jit('(float32[:], float32[:], int32, int32, float32[:])',
          device=True,
          inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    """line segment intersection"""
    a = cuda.local.array((2,), dtype=numba_float32)
    b = cuda.local.array((2,), dtype=numba_float32)
    c = cuda.local.array((2,), dtype=numba_float32)
    d = cuda.local.array((2,), dtype=numba_float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]
    ba0 = b[0] - a[0]
    ba1 = b[1] - a[1]
    da0 = d[0] - a[0]
    ca0 = c[0] - a[0]
    da1 = d[1] - a[1]
    ca1 = c[1] - a[1]
    acd = da1 * ca0 > ca1 * da0
    bcd = (d[1] - b[1]) * (c[0] - b[0]) > (c[1] - b[1]) * (d[0] - b[0])
    if acd != bcd:
        abc = ca1 * ba0 > ba1 * ca0
        abd = da1 * ba0 > ba1 * da0
        if abc != abd:
            dc0 = d[0] - c[0]
            dc1 = d[1] - c[1]
            abba = a[0] * b[1] - b[0] * a[1]
            cddc = c[0] * d[1] - d[0] * c[1]
            dh = ba1 * dc0 - ba0 * dc1
            dx = abba * dc0 - ba0 * cddc
            dy = abba * dc1 - ba1 * cddc
            temp_pts[0] = dx / dh
            temp_pts[1] = dy / dh
            return True
    return False


@cuda.jit('(float32, float32, float32[:])', device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    """point in quadrilateral"""
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap >= 0 and adad >= adap >= 0


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    """quadrilateral intersection"""
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2,), dtype=numba_float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    """rbbox to corners"""
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4,), dtype=numba_float32)
    corners_y = cuda.local.array((4,), dtype=numba_float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i +
                1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def inter(rbbox1, rbbox2):
    """inter"""
    corners1 = cuda.local.array((8,), dtype=numba_float32)
    corners2 = cuda.local.array((8,), dtype=numba_float32)
    intersection_corners = cuda.local.array((16,), dtype=numba_float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)

    return area(intersection_corners, num_intersection)


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def dev_rotate_iou_eval(rbox1, rbox2, criterion=-1):
    """dev_rotate_iou_eval"""
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    if criterion == 0:
        return area_inter / area1
    if criterion == 1:
        return area_inter / area2
    return area_inter


@cuda.jit('(int64, int64, float32[:], float32[:], float32[:], int32)',
          fastmath=False)
def rotate_iou_kernel_eval(n,
                           k,
                           dev_boxes,
                           dev_query_boxes,
                           dev_iou,
                           criterion=-1):
    """rotate iou kernel eval"""
    threads_per_block = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(n - row_start * threads_per_block, threads_per_block)
    col_size = min(k - col_start * threads_per_block, threads_per_block)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba_float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5,), dtype=numba_float32)

    dev_query_box_idx = threads_per_block * col_start + tx
    dev_box_idx = threads_per_block * row_start + tx
    if tx < col_size:
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if tx < row_size:
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threads_per_block * k + col_start * threads_per_block + tx * k + i
            dev_iou[offset] = dev_rotate_iou_eval(block_qboxes[i * 5:i * 5 + 5],
                                                  block_boxes[tx * 5:tx * 5 + 5],
                                                  criterion)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou running in gpu"""
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    n_boxes = boxes.shape[0]
    k_qboxes = query_boxes.shape[0]
    iou = np.zeros((n_boxes, k_qboxes), dtype=np.float32)
    if n_boxes == 0 or k_qboxes == 0:
        return iou
    threads_per_block = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(n_boxes, threads_per_block), div_up(k_qboxes, threads_per_block))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threads_per_block, stream](n_boxes, k_qboxes, boxes_dev,
                                                                         query_boxes_dev, iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)


@numba_jit
def get_thresholds(scores, num_gt, num_sample_pts=41):
    """get thresholds"""
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def _clean_gt_data(anno, current_cls_name, difficulty):
    """clean gt data"""
    min_height = [40, 25, 25]
    max_occlusion = [0, 1, 2]
    max_truncation = [0.15, 0.3, 0.5]
    num = len(anno['name'])
    num_valid = 0
    dc_bboxes, ignored = [], []

    for i in range(num):
        bbox = anno['bbox'][i]
        name = anno['name'][i].lower()
        height = abs(bbox[3] - bbox[1])
        if name == current_cls_name:
            valid_class = 1
        elif ((current_cls_name == "pedestrian" and name == "person_sitting")
              or (current_cls_name == "car" and name == "van")):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((anno["occluded"][i] > max_occlusion[difficulty])
                or (anno["truncated"][i] > max_truncation[difficulty])
                or (height <= min_height[difficulty])):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored.append(0)
            num_valid += 1
        elif valid_class == 0 or (ignore and (valid_class == 1)):
            ignored.append(1)
        else:
            ignored.append(-1)
        if anno["name"][i] == "DontCare":
            dc_bboxes.append(bbox)
    return num_valid, ignored, dc_bboxes


def _clean_dt_data(anno, current_cls_name, difficulty):
    """clean dt data"""
    min_height = [40, 25, 25]
    num = len(anno['name'])
    ignored = []
    for i in range(num):
        if anno["name"][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(anno["bbox"][i, 3] - anno["bbox"][i, 1])
        if height < min_height[difficulty]:
            ignored.append(1)
        elif valid_class == 1:
            ignored.append(0)
        else:
            ignored.append(-1)
    return ignored


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    """clean data"""
    class_names = ['car', 'pedestrian', 'cyclist', 'van',
                   'person_sitting', 'car', 'tractor', 'trailer']
    current_cls_name = class_names[current_class].lower()

    num_valid_gt, ignored_gt, dc_bboxes = _clean_gt_data(gt_anno,
                                                         current_cls_name,
                                                         difficulty)
    ignored_dt = _clean_dt_data(dt_anno, current_cls_name, difficulty)
    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba_jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    """image box overlap"""
    n_boxes = boxes.shape[0]
    k_qboxes = query_boxes.shape[0]
    overlaps = np.zeros((n_boxes, k_qboxes), dtype=boxes.dtype)
    for k in range(k_qboxes):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(n_boxes):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    """bev box overlap"""
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba_jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    """3d box overlap kernel"""
    # ONLY support overlap in CAMERA, not lider.
    n_boxes, k_qboxes = boxes.shape[0], qboxes.shape[0]
    for i in range(n_boxes):
        for j in range(k_qboxes):
            if rinc[i, j] > 0:
                iw = (min(boxes[i, 1], qboxes[j, 1]) -
                      max(boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    """3d box overlap"""
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba_jit(nopython=True)
def compute_statistics_jit(overlaps, gt_datas, dt_datas,
                           ignored_gt, ignored_det,
                           dc_bboxes, metric, min_overlap,
                           thresh=0., compute_fp=False, compute_aos=False):
    """compute statistics jit"""
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    # Using a large negative number to filter the cases with no detections
    # for counting False Positives
    no_detection = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = no_detection
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j] == -1 or assigned_detection[j] or ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if not compute_fp and overlap > min_overlap and dt_score > valid_detection:
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and overlap > min_overlap
                  and (overlap > max_overlap or assigned_ignored_det) and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and overlap > min_overlap
                  and (valid_detection == no_detection) and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if valid_detection == no_detection and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != no_detection
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != no_detection:
            # only a tp add a threshold.
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j] or ignored_det[j] == -1
                            or ignored_det[j] == 1 or ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    """get split parts"""
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    return [same_part] * num_part + [remain_num]


@numba_jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    """fused compute statistics"""
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def _get_parted_overlaps(gt_annos, dt_annos, split_parts, metric):
    """get overlaps parted"""
    parted_overlaps = []
    example_idx = 0
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)

            loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)

            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)

            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part

    return parted_overlaps


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    parted_overlaps = _get_parted_overlaps(gt_annos, dt_annos, split_parts, metric)
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num]
            )
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    """prepare data"""
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if np.array(dc_bboxes).shape[0] == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate([gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
            dontcares, total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=50):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. official:
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]]
            format: [metric, class]. choose one from matrix above.
        compute_aos: bool. compute aos or not
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    if len(gt_annos) != len(dt_annos):
        raise ValueError(
            f'Number of elements in ground-truth and detected annotations '
            f'lists must be equal, got {len(gt_annos)} and {len(dt_annos)}.'
        )
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    n_sample_pts = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, n_sample_pts])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, n_sample_pts])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, n_sample_pts])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(overlaps[i], gt_datas_list[i], dt_datas_list[i],
                                                  ignored_gts[i], ignored_dets[i],
                                                  dontcares[i], metric, min_overlap=min_overlap,
                                                  thresh=0.0, compute_fp=False)
                    _, _, _, _, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos
                    )
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict


def get_map(prec):
    """get map"""
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            difficultys=(0, 1, 2)):
    """do eval"""
    # min_overlaps: [num_minoverlap, metric, num_class]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    map_bbox = get_map(ret["precision"])
    map_aos = None
    if compute_aos:
        map_aos = get_map(ret["orientation"])
    ret = eval_class(gt_annos, dt_annos, current_classes,
                     difficultys, 1, min_overlaps)
    map_bev = get_map(ret["precision"])
    ret = eval_class(gt_annos, dt_annos, current_classes,
                     difficultys, 2, min_overlaps)
    map_3d = get_map(ret["precision"])
    return map_bbox, map_bev, map_3d, map_aos


def do_coco_style_eval(gt_annos, dt_annos, current_classes,
                       overlap_ranges, compute_aos):
    """do coco style eval"""
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            start, end, num = overlap_ranges[:, i, j]
            min_overlaps[:, i, j] = np.linspace(start, end, int(num))
    map_bbox, map_bev, map_3d, map_aos = do_eval(gt_annos, dt_annos, current_classes,
                                                 min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap]
    map_bbox = map_bbox.mean(-1)
    map_bev = map_bev.mean(-1)
    map_3d = map_3d.mean(-1)
    if map_aos is not None:
        map_aos = map_aos.mean(-1)
    return map_bbox, map_bev, map_3d, map_aos


def print_str(value, *arg, sstream=None):
    """print str"""
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def get_official_eval_result(gt_annos, dt_annos, current_classes, difficultys=(0, 1, 2), return_data=False):
    """get official eval result"""
    min_overlaps = np.array([[
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7]
    ]])
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'car',
        6: 'tractor',
        7: 'trailer',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = '        Easy   Mod    Hard\n'
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    map_bbox, map_bev, map_3d, map_aos = do_eval(gt_annos, dt_annos, current_classes,
                                                 min_overlaps, compute_aos, difficultys)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j]))
            )
            result += print_str((f"bbox AP:{map_bbox[j, 0, i]:.2f}, "
                                 f"{map_bbox[j, 1, i]:.2f}, "
                                 f"{map_bbox[j, 2, i]:.2f}"))
            result += print_str((f"bev  AP:{map_bev[j, 0, i]:.2f}, "
                                 f"{map_bev[j, 1, i]:.2f}, "
                                 f"{map_bev[j, 2, i]:.2f}"))
            result += print_str((f"3d   AP:{map_3d[j, 0, i]:.2f}, "
                                 f"{map_3d[j, 1, i]:.2f}, "
                                 f"{map_3d[j, 2, i]:.2f}"))
            if compute_aos:
                result += print_str((f"aos  AP:{map_aos[j, 0, i]:.2f}, "
                                     f"{map_aos[j, 1, i]:.2f}, "
                                     f"{map_aos[j, 2, i]:.2f}"))
    if return_data:
        return result, map_bbox, map_bev, map_3d, map_aos
    return result


def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    """get coco eval result"""
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'car',
        6: 'tractor',
        7: 'trailer',
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
        5: [0.5, 0.95, 10],
        6: [0.5, 0.95, 10],
        7: [0.5, 0.95, 10],

    }

    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(class_to_range[curcls])[:, np.newaxis]

    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    map_bbox, map_bev, map_3d, map_aos = do_coco_style_eval(gt_annos, dt_annos, current_classes,
                                                            overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str(
            (f"{class_to_name[curcls]} "
             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range))
        )
        result += print_str((f"bbox AP:{map_bbox[j, 0]:.2f}, "
                             f"{map_bbox[j, 1]:.2f}, "
                             f"{map_bbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{map_bev[j, 0]:.2f}, "
                             f"{map_bev[j, 1]:.2f}, "
                             f"{map_bev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{map_3d[j, 0]:.2f}, "
                             f"{map_3d[j, 1]:.2f}, "
                             f"{map_3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{map_aos[j, 0]:.2f}, "
                                 f"{map_aos[j, 1]:.2f}, "
                                 f"{map_aos[j, 2]:.2f}"))
    return result
