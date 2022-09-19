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
"""predict"""
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import numpy as mnp
from mindspore import ops

from src.core import box_ops
from src.core import nms
from src.data import kitti_common as kitti


def get_index_by_mask(mask):
    """get index by mask"""
    if isinstance(mask, np.ndarray):
        return np.where(mask)[0]
    return Tensor(np.where(mask.asnumpy()))[0]


def xor(a, b):
    """xor"""
    return Tensor(a.asnumpy() ^ b.asnumpy())


def _get_top_scores_labels(total_scores, num_class_with_bg, nms_score_threshold):
    """get top scores"""
    if num_class_with_bg == 1:
        top_scores = total_scores.squeeze(-1)
        if isinstance(top_scores, np.ndarray):
            top_labels = np.zeros(total_scores.shape[0], dtype=np.int64)
        else:
            top_labels = ops.Zeros()(total_scores.shape[0], mstype.int64)
    else:
        if isinstance(total_scores, np.ndarray):
            top_scores = np.max(total_scores, axis=-1)
            top_labels = np.argmax(total_scores, axis=-1)
        else:
            top_labels, top_scores = ops.ArgMaxWithValue(axis=-1)(total_scores)

    top_scores_keep = Tensor([])
    if nms_score_threshold > 0.0:
        if isinstance(top_scores, np.ndarray):   # for nms_score_thre type is np
            thresh = np.array([nms_score_threshold], dtype=total_scores.dtype)
            top_scores_keep = (top_scores >= thresh)
        else:
            thresh = Tensor([nms_score_threshold], dtype=total_scores.dtype)
            top_scores_keep = (top_scores >= thresh).astype(mstype.float16)
        if top_scores_keep.sum() > 0:
            if isinstance(top_scores, np.ndarray):
                top_scores = top_scores[top_scores_keep]
            else:
                top_scores_keep = get_index_by_mask(top_scores_keep)
                top_scores = top_scores[top_scores_keep]
        else:
            top_scores = Tensor([])

    return top_scores, top_labels, top_scores_keep


def _get_selected_data(total_scores,
                       box_preds,
                       dir_labels,
                       cfg):
    """get selected data"""
    selected_boxes = None
    selected_labels = None
    selected_scores = None
    selected_dir_labels = None

    # get highest score per prediction, then apply nms
    # to remove overlapped box.
    top_scores, top_labels, top_scores_keep = _get_top_scores_labels(total_scores,
                                                                     cfg['num_class_with_bg'],
                                                                     cfg['nms_score_threshold'])
    if top_scores.shape[0] != 0:
        if cfg['nms_score_threshold'] > 0.0:
            box_preds = box_preds[top_scores_keep]
            if cfg['use_direction_classifier']:
                dir_labels = dir_labels[top_scores_keep]
            top_labels = top_labels[top_scores_keep]
        boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]

        box_preds_corners = box_ops.center_to_corner_box2d(boxes_for_nms[:, :2],
                                                           boxes_for_nms[:, 2:4],
                                                           boxes_for_nms[:, 4])
        if isinstance(box_preds_corners, np.ndarray):
            boxes_for_nms = box_ops.corner_to_standup_nd_np(box_preds_corners)
            selected = nms.nms_np(boxes_for_nms,
                                  top_scores,
                                  pre_max_size=cfg['nms_pre_max_size'],
                                  post_max_size=cfg['nms_post_max_size'],
                                  iou_threshold=cfg['nms_iou_threshold'])
        else:
            boxes_for_nms = box_ops.corner_to_standup_nd(box_preds_corners)
            # the nms in 3d detection just remove overlap boxes.
            selected = nms.nms(boxes_for_nms,
                               top_scores,
                               pre_max_size=cfg['nms_pre_max_size'],
                               post_max_size=cfg['nms_post_max_size'],
                               iou_threshold=cfg['nms_iou_threshold'])
    else:
        selected = None
    if selected is not None:
        selected_boxes = box_preds[selected]
        selected_labels = top_labels[selected]
        selected_scores = top_scores[selected]
        if cfg['use_direction_classifier']:
            selected_dir_labels = dir_labels[selected]

    return selected_boxes, selected_labels, selected_scores, selected_dir_labels


def softmax(x, axis=1):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    prob_x = x_exp / x_sum
    return prob_x

def sigmoid(x):
    x = 1 + (1 / np.exp(x))
    prob_x = 1 / x
    return prob_x


def _get_total_scores(cls_preds, cfg):
    """get total scores"""
    if cfg['encode_background_as_zeros']:
        if isinstance(cls_preds, np.ndarray):
            total_scores = sigmoid(cls_preds)
        else:
            total_scores = ops.Sigmoid()(cls_preds)
    else:
        # encode background as first element in one-hot vector
        if cfg['use_sigmoid_score']:
            if isinstance(cls_preds, np.ndarray):
                total_scores = sigmoid(cls_preds)[..., 1:]
            else:
                total_scores = ops.Sigmoid()(cls_preds)[..., 1:]
        else:
            if isinstance(cls_preds, np.ndarray):
                total_scores = softmax(cls_preds, -1)[..., 1:]
            else:
                total_scores = ops.Softmax(axis=-1)(cls_preds)[..., 1:]

    return total_scores


def predict(example, preds_dict, cfg, box_coder):
    use_direction_classifier = cfg['use_direction_classifier']
    num_class = cfg['num_class']
    batch_size = example['anchors'].shape[0]
    if isinstance(example["anchors"], np.ndarray):
        batch_anchors = example["anchors"].reshape(batch_size, -1, 7)
    else:
        batch_anchors = example["anchors"].view(batch_size, -1, 7)
    batch_rect = example["rect"]
    batch_trv2c = example["Trv2c"]
    batch_p2 = example["P2"]
    if "anchors_mask" not in example:
        batch_anchors_mask = [None] * batch_size
    else:
        if isinstance(example["anchors_mask"], np.ndarray):
            batch_anchors_mask = example["anchors_mask"].reshape(batch_size, -1)
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
    batch_imgidx = example['image_idx']
    batch_box_preds = preds_dict["box_preds"]
    batch_cls_preds = preds_dict["cls_preds"]
    if isinstance(batch_box_preds, np.ndarray):
        batch_box_preds = batch_box_preds.reshape(batch_size, -1, box_coder.code_size)
    else:
        batch_box_preds = batch_box_preds.view(batch_size, -1, box_coder.code_size)
    num_class_with_bg = num_class
    if not cfg['encode_background_as_zeros']:
        num_class_with_bg = num_class + 1
    cfg['num_class_with_bg'] = num_class_with_bg
    if isinstance(batch_cls_preds, np.ndarray):
        batch_cls_preds = batch_cls_preds.reshape(batch_size, -1, num_class_with_bg)
    else:
        batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
    batch_box_preds = box_coder.decode(batch_box_preds, batch_anchors)
    if use_direction_classifier:
        batch_dir_preds = preds_dict["dir_cls_preds"]
        if isinstance(batch_cls_preds, np.ndarray):
            batch_dir_preds = batch_dir_preds.reshape(batch_size, -1, 2)
        else:
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
    else:
        batch_dir_preds = [None] * batch_size
    predictions_dicts = []
    for box_preds, cls_preds, dir_preds, rect, trv2c, p2, img_idx, a_mask in zip(
            batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
            batch_trv2c, batch_p2, batch_imgidx, batch_anchors_mask):
        dir_labels = None
        if a_mask is not None:
            a_mask = get_index_by_mask(a_mask)
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            if use_direction_classifier:
                dir_preds = dir_preds[a_mask]
                if isinstance(dir_preds, np.ndarray):
                    dir_labels = np.argmax(dir_preds, axis=-1)
                else:
                    dir_labels = ops.ArgMaxWithValue(axis=-1)(dir_preds)[0]
        total_scores = _get_total_scores(cls_preds, cfg)
        selected_data = _get_selected_data(total_scores, box_preds, dir_labels, cfg)
        selected_boxes, selected_labels, selected_scores, selected_dir_labels = selected_data
        if selected_boxes is not None:
            if use_direction_classifier:
                if isinstance(selected_boxes, np.ndarray):
                    opp_labels = (selected_boxes[..., -1] > 0) ^ selected_dir_labels.astype(np.bool_)
                else:
                    opp_labels = xor((selected_boxes[..., -1] > 0), selected_dir_labels)
                if isinstance(opp_labels, np.ndarray):
                    selected_boxes[..., -1] += np.where(opp_labels, np.pi, 0.)
                else:
                    selected_boxes[..., -1] += mnp.where(opp_labels, Tensor(mnp.pi, dtype=selected_boxes.dtype),
                                                         Tensor(0.0, dtype=selected_boxes.dtype))
            final_box_preds_camera = box_ops.box_lidar_to_camera(selected_boxes, rect, trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_ops.center_to_corner_box3d(locs, dims, angles, camera_box_origin, axis=1)
            if isinstance(box_corners, np.ndarray):
                box_corners_in_image = box_ops.project_to_image_np(box_corners, p2)
                minxy = np.min(box_corners_in_image, axis=1)
                maxxy = np.max(box_corners_in_image, axis=1)
                box_2d_preds = np.concatenate([minxy, maxxy], axis=1)
            else:
                box_corners_in_image = box_ops.project_to_image(box_corners, p2)
                minxy = ops.ArgMinWithValue(axis=1)(box_corners_in_image)[1]
                maxxy = ops.ArgMaxWithValue(axis=1)(box_corners_in_image)[1]
                box_2d_preds = ops.Concat(axis=1)([minxy, maxxy])
            predictions_dict = {
                "bbox": box_2d_preds, "box3d_camera": final_box_preds_camera, "box3d_lidar": selected_boxes,
                "scores": selected_scores, "label_preds": selected_labels, "image_idx": img_idx,
            }
        else:
            predictions_dict = {
                "bbox": None, "box3d_camera": None, "box3d_lidar": None, "scores": None, "label_preds": None,
                "image_idx": img_idx,
            }
        predictions_dicts.append(predictions_dict)
    return predictions_dicts


def predict_kitti_to_anno(predictions_dicts,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False):
    """predict kitti to anno"""
    if isinstance(example['image_shape'], np.ndarray):
        batch_image_shape = example['image_shape']
    else:
        batch_image_shape = example['image_shape'].asnumpy()
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        for k, v in preds_dict.items():
            if v is not None:
                # if isinstance(v, np.int64):
                if isinstance(example['image_shape'], np.ndarray):
                    preds_dict[k] = v
                else:
                    preds_dict[k] = v.asnumpy()
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"]
            box_preds = preds_dict["box3d_camera"]
            scores = preds_dict["scores"]
            box_preds_lidar = preds_dict["box3d_lidar"]
            # write pred to file
            label_preds = preds_dict["label_preds"]
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(box_preds, box_preds_lidar,
                                                          box_2d_preds, scores, label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
    return annos
