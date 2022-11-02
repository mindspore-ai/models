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
"""eval script"""

import os
import argparse
from datetime import datetime
import logging
import re
import glob
import tqdm
import numpy as np

import mindspore as ms
from mindspore import ops

from src.lib.net.point_rcnn import PointRCNN
from src.lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from src.datautil import batchpad
import src.train_utils.train_utils as train_utils
from src.lib.utils.bbox_transform import decode_bbox_target
from src.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate

from src.lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import src.lib.utils.kitti_utils as kitti_utils
import src.lib.utils.iou3d.iou3d_utils as iou3d_utils

np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description="evaluate PointRCNN Model")
parser.add_argument('--cfg_file',
                    type=str,
                    default='config/default.yml',
                    help='specify the config for evaluation')
parser.add_argument("--eval_mode",
                    type=str,
                    default='rpn',
                    required=True,
                    help="specify the evaluation mode")

parser.add_argument('--eval_all',
                    action='store_true',
                    default=False,
                    help='whether to evaluate all checkpoints')
parser.add_argument('--test',
                    action='store_true',
                    default=False,
                    help='evaluate without ground truth')
parser.add_argument("--ckpt",
                    type=str,
                    default=None,
                    help="specify a checkpoint to be evaluated")
parser.add_argument("--rpn_ckpt",
                    type=str,
                    default=None,
                    help="specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt",
                    type=str,
                    default=None,
                    help="specify the checkpoint of rcnn if trained separated")

parser.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help='batch size for evaluation')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of workers for dataloader')
parser.add_argument("--extra_tag",
                    type=str,
                    default='default',
                    help="extra tag for multiple evaluation")
parser.add_argument('--output_dir',
                    type=str,
                    default=None,
                    help='specify an output directory if needed')
parser.add_argument("--ckpt_dir",
                    type=str,
                    default=None,
                    help="specify a ckpt directory to be evaluated if needed")

parser.add_argument('--save_result',
                    action='store_true',
                    default=False,
                    help='save evaluation results to files')
parser.add_argument(
    '--save_rpn_feature',
    action='store_true',
    default=False,
    help='save features for separately rcnn training and evaluation')

parser.add_argument('--random_select',
                    action='store_true',
                    default=True,
                    help='sample to the same number of points')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='ignore the checkpoint smaller than this epoch')
parser.add_argument(
    "--rcnn_eval_roi_dir",
    type=str,
    default=None,
    help=
    'specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument(
    "--rcnn_eval_feature_dir",
    type=str,
    default=None,
    help=
    'specify the saved features for rcnn evaluation when using rcnn_offline mode'
)
parser.add_argument('--set',
                    dest='set_cfgs',
                    default=None,
                    nargs=argparse.REMAINDER,
                    help='set extra config keys if needed')
args = parser.parse_args()


def create_logger(log_file):
    """create logger"""
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def save_kitti_format(sample_id, calib, bbox3d, kitti_output_dir, scores,
                      img_shape):
    """save kitti format"""
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8,
                                    img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print(
                '%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                % (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1],
                   img_boxes[k, 2], img_boxes[k, 3], bbox3d[k, 3],
                   bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1],
                   bbox3d[k, 2], bbox3d[k, 6], scores[k]),
                file=f)


def save_rpn_features(seg_result, rpn_scores_raw, pts_features, backbone_xyz,
                      backbone_features, kitti_features_dir, sample_id):
    """save rpn features"""
    pts_intensity = pts_features[:, 0]

    output_file = os.path.join(kitti_features_dir, '%06d.npy' % sample_id)
    xyz_file = os.path.join(kitti_features_dir, '%06d_xyz.npy' % sample_id)
    seg_file = os.path.join(kitti_features_dir, '%06d_seg.npy' % sample_id)
    intensity_file = os.path.join(kitti_features_dir,
                                  '%06d_intensity.npy' % sample_id)
    np.save(output_file, backbone_features)
    np.save(xyz_file, backbone_xyz)
    np.save(seg_file, seg_result)
    np.save(intensity_file, pts_intensity)
    rpn_scores_raw_file = os.path.join(kitti_features_dir,
                                       '%06d_rawscore.npy' % sample_id)
    np.save(rpn_scores_raw_file, rpn_scores_raw)


def make_dirs(result_dir):
    """make directories"""
    if args.save_rpn_feature:
        kitti_features_dir = os.path.join(result_dir, 'features')
        os.makedirs(kitti_features_dir, exist_ok=True)

    if args.save_result or args.save_rpn_feature:
        kitti_output_dir = os.path.join(result_dir, 'detections', 'data')
        seg_output_dir = os.path.join(result_dir, 'seg_result')
        os.makedirs(kitti_output_dir, exist_ok=True)
        os.makedirs(seg_output_dir, exist_ok=True)

    return kitti_features_dir, kitti_output_dir, seg_output_dir


def inference(pts_input, model):
    """model inference"""
    inputs = ms.Tensor.from_numpy(pts_input).astype(ms.float32)
    input_data = {'pts_input': inputs}

    # model inference
    ret_dict = model(input_data)
    rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
    backbone_xyz, backbone_features = ret_dict['backbone_xyz'], ret_dict[
        'backbone_features']

    rpn_scores_raw = rpn_cls[:, :, 0]
    rpn_scores = ops.sigmoid(rpn_scores_raw)
    seg_result = (rpn_scores > cfg.RPN.SCORE_THRESH).astype(ms.int32)

    # proposal layer
    rois, roi_scores_raw = model.rpn.proposal_layer(rpn_scores_raw, rpn_reg,
                                                    backbone_xyz)  # (B, M, 7)
    out_dict = {
        'backbone_xyz': backbone_xyz,
        'backbone_features': backbone_features,
        'seg_result': seg_result,
        'rois': rois,
        'rpn_scores_raw': rpn_scores_raw,
        'roi_scores_raw': roi_scores_raw
    }
    return out_dict


def get_gt_data(data):
    """get groundtruth data"""
    rpn_cls_label, _ = data['rpn_cls_label'], data['rpn_reg_label']
    gt_boxes3d = data['gt_boxes3d']

    rpn_cls_label = ms.Tensor.from_numpy(rpn_cls_label).astype(ms.int32)
    if gt_boxes3d.shape[1] == 0:  # (B, M, 7)
        pass
    else:
        gt_boxes3d = ms.Tensor.from_numpy(gt_boxes3d).astype(ms.float32)
    return rpn_cls_label, gt_boxes3d


def eval_one_epoch_rpn(model, dataloader, epoch_id, result_dir, logger):
    """eval one epocn for rpn"""
    np.random.seed(1024)
    mode = 'TEST' if args.test else 'EVAL'

    kitti_features_dir, kitti_output_dir, seg_output_dir = make_dirs(
        result_dir)

    logger.info('---- EPOCH %s RPN EVALUATION ----' % epoch_id)

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    dataset = dataloader.dataset
    cnt = max_num = rpn_iou_avg = 0

    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval')

    for data in dataloader:
        sample_id_list, pts_rect, pts_features, pts_input = \
            data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input']
        cnt += len(sample_id_list)

        if not args.test:
            rpn_cls_label, gt_boxes3d = get_gt_data(data)

        out_dict = inference(pts_input, model)
        backbone_xyz, backbone_features, seg_result, rois, rpn_scores_raw, roi_scores_raw = out_dict.values(
        )

        # calculate recall and save results to file
        for bs_idx in range(rois.shape[0]):
            cur_sample_id = sample_id_list[bs_idx]
            cur_scores_raw = roi_scores_raw[bs_idx]  # (N)
            cur_boxes3d = rois[bs_idx]  # (N, 7)
            cur_seg_result = seg_result[bs_idx]
            cur_pts_rect = pts_rect[bs_idx]

            # calculate recall
            if not args.test:
                cur_rpn_cls_label = rpn_cls_label[bs_idx]
                cur_gt_boxes3d = gt_boxes3d[bs_idx]

                k = len(cur_gt_boxes3d) - 1
                while k > 0 and cur_gt_boxes3d[k].sum() == 0:
                    k -= 1
                cur_gt_boxes3d = cur_gt_boxes3d[:k + 1]

                if cur_gt_boxes3d.shape[0] > 0:
                    iou3d = iou3d_utils.boxes_iou3d_gpu(
                        cur_boxes3d, cur_gt_boxes3d[:, 0:7])
                    gt_max_iou, _ = iou3d.max(axis=0)

                    for idx, thresh in enumerate(thresh_list):
                        total_recalled_bbox_list[idx] += (
                            gt_max_iou > thresh).sum().asnumpy().item()
                    total_gt_bbox += len(cur_gt_boxes3d)

                fg_mask = cur_rpn_cls_label > 0
                correct = ((cur_seg_result == cur_rpn_cls_label)
                           & fg_mask).sum().astype(ms.float32)
                union = fg_mask.sum().astype(ms.float32) + (
                    cur_seg_result > 0).sum().astype(ms.float32) - correct

                rpn_iou = correct / ops.clip_by_value(
                    union, min=ms.Tensor(1.0, ms.float32))
                rpn_iou_avg += rpn_iou.asnumpy().asnumpy().item()

            # save result
            if args.save_rpn_feature:
                # save features to file
                save_rpn_features(
                    seg_result[bs_idx].astype(ms.float32).asnumpy(),
                    rpn_scores_raw[bs_idx].astype(ms.float32).asnumpy(),
                    pts_features[bs_idx], backbone_xyz[bs_idx].asnumpy(),
                    backbone_features[bs_idx].asnumpy().transpose(1, 0),
                    kitti_features_dir, cur_sample_id)

            if args.save_result or args.save_rpn_feature:
                cur_pred_cls = cur_seg_result.asnumpy()
                output_file = os.path.join(seg_output_dir,
                                           '%06d.npy' % cur_sample_id)
                if not args.test:
                    cur_gt_cls = cur_rpn_cls_label.asnumpy()
                    output_data = np.concatenate(
                        (cur_pts_rect.reshape(-1, 3), cur_gt_cls.reshape(
                            -1, 1), cur_pred_cls.reshape(-1, 1)),
                        axis=1)
                else:
                    output_data = np.concatenate((cur_pts_rect.reshape(
                        -1, 3), cur_pred_cls.reshape(-1, 1)),
                                                 axis=1)

                np.save(output_file, output_data.astype(np.float16))

                # save as kitti format
                calib = dataset.get_calib(cur_sample_id)
                cur_boxes3d = cur_boxes3d.asnumpy()
                image_shape = dataset.get_image_shape(cur_sample_id)
                save_kitti_format(cur_sample_id, calib, cur_boxes3d,
                                  kitti_output_dir, cur_scores_raw,
                                  image_shape)

        disp_dict = {
            'mode': mode,
            'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox),
            'rpn_iou': rpn_iou_avg / max(cnt, 1.0)
        }
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

    progress_bar.close()
    res_info = {
        'max_num': max_num,
        'rpn_iou_avg': rpn_iou_avg,
        'cnt': cnt,
        'total_gt_bbox': total_gt_bbox,
        'result_dir': result_dir
    }

    ret_dict = show_log(logger, epoch_id, thresh_list,
                        total_recalled_bbox_list, res_info)

    return ret_dict


def show_log(logger, epoch_id, thresh_list, total_recalled_bbox_list,
             res_info):
    """show log"""

    logger.info(str(datetime.now()))
    logger.info(
        '-------------------performance of epoch %s---------------------' %
        epoch_id)
    logger.info('max number of objects: %d' % res_info['max_num'])
    logger.info('rpn iou avg: %f' %
                (res_info['rpn_iou_avg'] / max(res_info['cnt'], 1.0)))

    ret_dict = {
        'max_obj_num': res_info['max_num'],
        'rpn_iou': res_info['rpn_iou_avg'] / res_info['cnt']
    }

    for idx, thresh in enumerate(thresh_list):
        cur_recall = total_recalled_bbox_list[idx] / max(
            res_info['total_gt_bbox'], 1.0)
        logger.info('total bbox recall(thresh=%.3f): %d / %d = %f' %
                    (thresh, total_recalled_bbox_list[idx],
                     res_info['total_gt_bbox'], cur_recall))
        ret_dict['rpn_recall(thresh=%.2f)' % thresh] = cur_recall
    logger.info('result is saved to: %s' % res_info['result_dir'])

    return ret_dict


def eval_one_epoch_rcnn(model, dataloader, dataset, epoch_id, result_dir,
                        logger):
    """eval one epoch for rcnn"""
    np.random.seed(1024)
    MEAN_SIZE = ms.Tensor.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    mode = 'TEST' if args.test else 'EVAL'

    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)

    if args.save_result:
        roi_output_dir = os.path.join(result_dir, 'roi_result', 'data')
        refine_output_dir = os.path.join(result_dir, 'refine_result', 'data')
        os.makedirs(roi_output_dir, exist_ok=True)
        os.makedirs(refine_output_dir, exist_ok=True)

    logger.info('---- EPOCH %s RCNN EVALUATION ----' % epoch_id)
    model.eval()

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    cnt = final_total = total_cls_acc = total_cls_acc_refined = 0
    dataloader: ms.dataset.BatchDataset = dataloader

    progress_bar = tqdm.tqdm(total=dataloader.get_dataset_size(),
                             leave=True,
                             desc='eval')
    for data in dataloader.create_dict_iterator():
        sample_id = data['sample_id']
        cnt += 1
        assert args.batch_size == 1, 'Only support bs=1 here'
        input_data = {}
        for key, val in data.items():
            if key != 'sample_id':
                input_data[
                    key] = val

        roi_boxes3d = input_data['roi_boxes3d']
        roi_scores = input_data['roi_scores']
        if cfg.RCNN.ROI_SAMPLE_JIT:
            for key, val in input_data.items():
                if key in ['gt_iou', 'gt_boxes3d']:
                    continue
                input_data[key] = input_data[key].expand_dims(axis=0)
        else:

            pts_input = ops.concat(
                (input_data['pts_input'], input_data['pts_features']), axis=-1)
            input_data['pts_input'] = pts_input

        ret_dict = model(input_data)
        rcnn_cls = ret_dict['rcnn_cls']
        rcnn_reg = ret_dict['rcnn_reg']

        # bounding box regression
        anchor_size = MEAN_SIZE
        if cfg.RCNN.SIZE_RES_ON_ROI:
            roi_size = input_data['roi_size']
            anchor_size = roi_size

        pred_boxes3d = decode_bbox_target(
            roi_boxes3d,
            rcnn_reg,
            anchor_size=anchor_size,
            loc_scope=cfg.RCNN.LOC_SCOPE,
            loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
            num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
            get_xz_fine=True,
            get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
            loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
            loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
            get_ry_fine=True)

        # scoring
        if rcnn_cls.shape[1] == 1:
            raw_scores = rcnn_cls.view(-1)

            norm_scores = ops.Sigmoid()(raw_scores)
            pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).astype(
                ms.int32)
        else:
            pred_classes = ops.Argmax(axis=1)(rcnn_cls).view(-1)

            cls_norm_scores = ops.Softmax(axis=1)(rcnn_cls)
            raw_scores = rcnn_cls[:, pred_classes]
            norm_scores = cls_norm_scores[:, pred_classes]

        # evaluation
        disp_dict = {'mode': mode}
        if not args.test:
            gt_boxes3d = input_data['gt_boxes3d']
            gt_iou = input_data['gt_iou']

            # calculate recall
            gt_num = gt_boxes3d.shape[0]
            if gt_num > 0:
                iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d, gt_boxes3d)
                gt_max_iou, _ = iou3d.max(axis=0)

                for idx, thresh in enumerate(thresh_list):
                    total_recalled_bbox_list[idx] += (
                        gt_max_iou > thresh).sum().asnumpy().item()
                total_gt_bbox += gt_num

                iou3d_in = iou3d_utils.boxes_iou3d_gpu(roi_boxes3d, gt_boxes3d)
                gt_max_iou_in, _ = iou3d_in.max(axis=0)

                for idx, thresh in enumerate(thresh_list):
                    total_roi_recalled_bbox_list[idx] += (
                        gt_max_iou_in > thresh).sum().asnumpy().item()

            # classification accuracy
            cls_label = (gt_iou > cfg.RCNN.CLS_FG_THRESH).astype(ms.float32)
            cls_valid_mask = ((gt_iou >= cfg.RCNN.CLS_FG_THRESH) |
                              (gt_iou <= cfg.RCNN.CLS_BG_THRESH)).astype(
                                  ms.float32)
            cls_acc = (
                (pred_classes == cls_label.astype(ms.int32)).astype(ms.float32)
                * cls_valid_mask).sum() / max(cls_valid_mask.sum(), 1.0)

            iou_thresh = 0.7 if cfg.CLASSES == 'Car' else 0.5
            cls_label_refined = (gt_iou >= iou_thresh).astype(ms.float32)
            cls_acc_refined = (pred_classes == cls_label_refined.astype(
                ms.int32)).astype(ms.float32).sum() / max(
                    cls_label_refined.shape[0], 1.0)

            total_cls_acc += cls_acc.asnumpy().item()
            total_cls_acc_refined += cls_acc_refined.asnumpy().item()

            disp_dict['recall'] = '%d/%d' % (total_recalled_bbox_list[3],
                                             total_gt_bbox)
            disp_dict['cls_acc_refined'] = '%.2f' % cls_acc_refined.asnumpy(
            ).item()

        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

        image_shape = dataset.get_image_shape(sample_id)
        if args.save_result:
            # save roi and refine results
            roi_boxes3d_np = roi_boxes3d.asnumpy()
            pred_boxes3d_np = pred_boxes3d.asnumpy()
            calib = dataset.get_calib(sample_id)

            save_kitti_format(sample_id, calib, roi_boxes3d_np, roi_output_dir,
                              roi_scores, image_shape)
            save_kitti_format(sample_id, calib,
                              pred_boxes3d_np, refine_output_dir,
                              raw_scores.asnumpy(), image_shape)

        # NMS and scoring
        # scores thresh
        inds = norm_scores > cfg.RCNN.SCORE_THRESH
        if inds.sum() == 0:
            continue

        pred_boxes3d_selected = pred_boxes3d[inds]
        raw_scores_selected = raw_scores[inds]

        # NMS thresh
        boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(
            pred_boxes3d_selected)
        keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected,
                                       cfg.RCNN.NMS_THRESH)
        pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]

        scores_selected = raw_scores_selected[keep_idx]
        pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.asnumpy(
        ), scores_selected.asnumpy()

        calib = dataset.get_calib(sample_id)
        final_total += pred_boxes3d_selected.shape[0]
        save_kitti_format(sample_id, calib, pred_boxes3d_selected,
                          final_output_dir, scores_selected, image_shape)

    progress_bar.close()

    # dump empty files
    split_file = os.path.join(dataset.imageset_dir, '..', '..', 'ImageSets',
                              dataset.split + '.txt')
    split_file = os.path.abspath(split_file)
    image_idx_list = [x.strip() for x in open(split_file).readlines()]
    empty_cnt = 0
    for k in range(len(image_idx_list)):
        cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
        if not os.path.exists(cur_file):
            empty_cnt += 1
            logger.info('empty_cnt=%d: dump empty file %s' %
                        (empty_cnt, cur_file))

    ret_dict = {'empty_cnt': empty_cnt}

    logger.info(
        '-------------------performance of epoch %s---------------------' %
        epoch_id)
    logger.info(str(datetime.now()))

    avg_cls_acc = (total_cls_acc / max(cnt, 1.0))
    avg_cls_acc_refined = (total_cls_acc_refined / max(cnt, 1.0))
    avg_det_num = (final_total / max(cnt, 1.0))
    logger.info('final average detections: %.3f' % avg_det_num)
    logger.info('final average cls acc: %.3f' % avg_cls_acc)
    logger.info('final average cls acc refined: %.3f' % avg_cls_acc_refined)
    ret_dict['rcnn_cls_acc'] = avg_cls_acc
    ret_dict['rcnn_cls_acc_refined'] = avg_cls_acc_refined
    ret_dict['rcnn_avg_num'] = avg_det_num

    for idx, thresh in enumerate(thresh_list):
        cur_roi_recall = total_roi_recalled_bbox_list[idx] / max(
            total_gt_bbox, 1.0)
        logger.info('total roi bbox recall(thresh=%.3f): %d / %d = %f' %
                    (thresh, total_roi_recalled_bbox_list[idx], total_gt_bbox,
                     cur_roi_recall))
        ret_dict['rpn_recall(thresh=%.2f)' % thresh] = cur_roi_recall

    for idx, thresh in enumerate(thresh_list):
        cur_recall = total_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
        logger.info(
            'total bbox recall(thresh=%.3f): %d / %d = %f' %
            (thresh, total_recalled_bbox_list[idx], total_gt_bbox, cur_recall))
        ret_dict['rcnn_recall(thresh=%.2f)' % thresh] = cur_recall

    if cfg.TEST.SPLIT != 'test':
        logger.info('Averate Precision:')
        name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        ap_result_str, ap_dict = kitti_evaluate(
            dataset.label_dir,
            final_output_dir,
            label_split_file=split_file,
            current_class=name_to_class[cfg.CLASSES])
        logger.info(ap_result_str)
        ret_dict.update(ap_dict)

    logger.info('result is saved to: %s' % result_dir)

    return ret_dict


def eval_one_epoch_joint(model, dataloader, dataset, epoch_id, result_dir,
                         logger):
    """eval one epoch joint"""
    MEAN_SIZE = ms.Tensor.from_numpy(cfg.CLS_MEAN_SIZE[0])
    mode = 'TEST' if args.test else 'EVAL'

    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)

    if args.save_result:
        roi_output_dir = os.path.join(result_dir, 'roi_result', 'data')
        refine_output_dir = os.path.join(result_dir, 'refine_result', 'data')
        rpn_output_dir = os.path.join(result_dir, 'rpn_result', 'data')
        os.makedirs(rpn_output_dir, exist_ok=True)
        os.makedirs(roi_output_dir, exist_ok=True)
        os.makedirs(refine_output_dir, exist_ok=True)

    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch_id)
    logger.info('==> Output file: %s' % result_dir)

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0
    dataloader: ms.dataset.BatchDataset = dataloader

    progress_bar = tqdm.tqdm(total=dataloader.get_dataset_size(),
                             leave=True,
                             desc='eval')
    it = dataloader.create_dict_iterator()
    for data in it:
        cnt += 1
        sample_id, _, _, pts_input = \
            data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input']
        batch_size = len(sample_id)
        inputs = pts_input.astype(ms.float32)
        input_data = {'pts_input': inputs}

        # model inference
        ret_dict = model(input_data)

        roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
        roi_boxes3d = ret_dict['rois']  # (B, M, 7)
        seg_result = ret_dict['seg_result'].astype(ms.int32)  # (B, N)

        rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1,
                                             ret_dict['rcnn_cls'].shape[1])
        rcnn_reg = ret_dict['rcnn_reg'].view(
            batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

        # bounding box regression
        anchor_size = MEAN_SIZE
        if cfg.RCNN.SIZE_RES_ON_ROI:
            assert False

        pred_boxes3d = decode_bbox_target(
            roi_boxes3d.view(-1, 7),
            rcnn_reg.view(-1, rcnn_reg.shape[-1]),
            anchor_size=anchor_size,
            loc_scope=cfg.RCNN.LOC_SCOPE,
            loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
            num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
            get_xz_fine=True,
            get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
            loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
            loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
            get_ry_fine=True).view(batch_size, -1, 7)

        # scoring
        if rcnn_cls.shape[2] == 1:
            raw_scores = rcnn_cls  # (B, M, 1)

            norm_scores = ops.Sigmoid()(raw_scores)
            pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).astype(
                ms.int32)
        else:
            pred_classes = ops.Argmax(axis=1)(rcnn_cls).view(-1)
            cls_norm_scores = ops.Softmax(axis=1)(rcnn_cls)
            raw_scores = rcnn_cls[:, pred_classes]
            norm_scores = cls_norm_scores[:, pred_classes]

        # evaluation
        recalled_num = gt_num = rpn_iou = 0
        if not args.test:
            if not cfg.RPN.FIXED:
                rpn_cls_label, _ = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = ms.Tensor.from_numpy(rpn_cls_label).astype(
                    ms.int32)

            gt_boxes3d = data['gt_boxes3d']

            for k in range(batch_size):
                # calculate recall
                cur_gt_boxes3d = gt_boxes3d[k]
                tmp_idx = len(cur_gt_boxes3d) - 1

                while tmp_idx >= 0 and cur_gt_boxes3d[tmp_idx].sum() == 0:
                    tmp_idx -= 1

                if tmp_idx >= 0:
                    cur_gt_boxes3d = cur_gt_boxes3d[:tmp_idx + 1]

                    cur_gt_boxes3d = cur_gt_boxes3d.astype(ms.float32)
                    iou3d = iou3d_utils.boxes_iou3d_gpu(
                        pred_boxes3d[k], cur_gt_boxes3d)
                    ops.ArgMaxWithValue(0)(iou3d)
                    _, gt_max_iou = ops.ArgMaxWithValue(0)(iou3d)

                    for idx, thresh in enumerate(thresh_list):
                        total_recalled_bbox_list[idx] += (
                            gt_max_iou > thresh).sum().asnumpy().item()
                    recalled_num += (gt_max_iou > 0.7).sum().asnumpy().item()
                    gt_num += cur_gt_boxes3d.shape[0]
                    total_gt_bbox += cur_gt_boxes3d.shape[0]

                    # original recall
                    iou3d_in = iou3d_utils.boxes_iou3d_gpu(
                        roi_boxes3d[k], cur_gt_boxes3d)
                    _, gt_max_iou_in = ops.ArgMaxWithValue(0)(iou3d_in)

                    for idx, thresh in enumerate(thresh_list):
                        total_roi_recalled_bbox_list[idx] += (
                            gt_max_iou_in > thresh).sum().asnumpy().item()

                if not cfg.RPN.FIXED:
                    fg_mask = rpn_cls_label > 0
                    correct = ((seg_result == rpn_cls_label)
                               & fg_mask).sum().astype(ms.float32)
                    union = fg_mask.sum().astype(ms.float32) + (
                        seg_result > 0).sum().astype(ms.float32) - correct
                    rpn_iou = correct / ops.clip_by_value(
                        union, min=ms.Tensor(1.0, ms.float32))
                    total_rpn_iou += rpn_iou.asnumpy().item()

        disp_dict = {
            'mode': mode,
            'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox)
        }
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

        if args.save_result:
            # save roi and refine results
            roi_boxes3d_np = roi_boxes3d.asnumpy()
            pred_boxes3d_np = pred_boxes3d.asnumpy()
            roi_scores_raw_np = roi_scores_raw.asnumpy()
            raw_scores_np = raw_scores.asnumpy()

            rpn_cls_np = ret_dict['rpn_cls'].asnumpy()
            rpn_xyz_np = ret_dict['backbone_xyz'].asnumpy()
            seg_result_np = seg_result.asnumpy()
            output_data = np.concatenate(
                (rpn_xyz_np, rpn_cls_np.reshape(batch_size, -1, 1),
                 seg_result_np.reshape(batch_size, -1, 1)),
                axis=2)

            for k in range(batch_size):
                cur_sample_id = sample_id[k]
                calib = dataset.get_calib(cur_sample_id)
                image_shape = dataset.get_image_shape(cur_sample_id)
                save_kitti_format(cur_sample_id, calib, roi_boxes3d_np[k],
                                  roi_output_dir, roi_scores_raw_np[k],
                                  image_shape)
                save_kitti_format(cur_sample_id, calib, pred_boxes3d_np[k],
                                  refine_output_dir, raw_scores_np[k],
                                  image_shape)

                output_file = os.path.join(rpn_output_dir,
                                           '%06d.npy' % cur_sample_id)
                np.save(output_file, output_data.astype(np.float32))

        # scores thresh
        inds = norm_scores > cfg.RCNN.SCORE_THRESH

        for k in range(batch_size):
            cur_inds = inds[k].view(-1)
            if cur_inds.sum() == 0:
                continue

            temp = pred_boxes3d.shape[-1]
            pred_boxes3d_selected = pred_boxes3d[k].masked_select(
                cur_inds.expand_dims(-1)).reshape((-1, temp))
            temp = pred_boxes3d.shape[-1]
            raw_scores_selected = raw_scores[k].masked_select(
                cur_inds.expand_dims(-1))
            temp = pred_boxes3d.shape[-1]

            # NMS thresh
            # rotated nms
            boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(
                pred_boxes3d_selected)
            boxes_bev_selected = ms.Tensor.from_numpy(
                boxes_bev_selected.asnumpy())
            raw_scores_selected = ms.Tensor.from_numpy(
                raw_scores_selected.asnumpy())
            keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected,
                                           raw_scores_selected,
                                           cfg.RCNN.NMS_THRESH).view(-1)
            pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
            scores_selected = raw_scores_selected[keep_idx]
            pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.asnumpy(
            ), scores_selected.asnumpy()

            cur_sample_id = sample_id[k]
            calib = dataset.get_calib(cur_sample_id)
            final_total += pred_boxes3d_selected.shape[0]
            image_shape = dataset.get_image_shape(cur_sample_id)
            save_kitti_format(cur_sample_id, calib, pred_boxes3d_selected,
                              final_output_dir, scores_selected, image_shape)

    progress_bar.close()
    # dump empty files
    split_file = os.path.join(dataset.imageset_dir, '..', '..', 'ImageSets',
                              dataset.split + '.txt')
    split_file = os.path.abspath(split_file)
    image_idx_list = [x.strip() for x in open(split_file).readlines()]
    empty_cnt = 0
    for k in range(len(image_idx_list)):
        cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
        if not os.path.exists(cur_file):

            empty_cnt += 1
            logger.info('empty_cnt=%d: dump empty file %s' %
                        (empty_cnt, cur_file))

    ret_dict = {'empty_cnt': empty_cnt}

    logger.info(
        '-------------------performance of epoch %s---------------------' %
        epoch_id)
    logger.info(str(datetime.now()))

    avg_rpn_iou = (total_rpn_iou / max(cnt, 1.0))
    avg_cls_acc = (total_cls_acc / max(cnt, 1.0))
    avg_cls_acc_refined = (total_cls_acc_refined / max(cnt, 1.0))
    avg_det_num = (final_total / max(len(dataset), 1.0))
    logger.info('final average detections: %.3f' % avg_det_num)
    logger.info('final average rpn_iou refined: %.3f' % avg_rpn_iou)
    logger.info('final average cls acc: %.3f' % avg_cls_acc)
    logger.info('final average cls acc refined: %.3f' % avg_cls_acc_refined)
    ret_dict['rpn_iou'] = avg_rpn_iou
    ret_dict['rcnn_cls_acc'] = avg_cls_acc
    ret_dict['rcnn_cls_acc_refined'] = avg_cls_acc_refined
    ret_dict['rcnn_avg_num'] = avg_det_num

    for idx, thresh in enumerate(thresh_list):
        cur_roi_recall = total_roi_recalled_bbox_list[idx] / max(
            total_gt_bbox, 1.0)
        logger.info('total roi bbox recall(thresh=%.3f): %d / %d = %f' %
                    (thresh, total_roi_recalled_bbox_list[idx], total_gt_bbox,
                     cur_roi_recall))
        ret_dict['rpn_recall(thresh=%.2f)' % thresh] = cur_roi_recall

    for idx, thresh in enumerate(thresh_list):
        cur_recall = total_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
        logger.info(
            'total bbox recall(thresh=%.3f): %d / %d = %f' %
            (thresh, total_recalled_bbox_list[idx], total_gt_bbox, cur_recall))
        ret_dict['rcnn_recall(thresh=%.2f)' % thresh] = cur_recall

    if cfg.TEST.SPLIT != 'test':
        logger.info('Averate Precision:')
        name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        ap_result_str, ap_dict = kitti_evaluate(
            dataset.label_dir,
            final_output_dir,
            label_split_file=split_file,
            current_class=name_to_class[cfg.CLASSES])
        logger.info(ap_result_str)
        ret_dict.update(ap_dict)

    logger.info('result is saved to: %s' % result_dir)
    return ret_dict


def eval_one_epoch(model, dataloader, dataset, epoch_id, result_dir, logger):
    """eval one epoch"""
    if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_rpn(model, dataloader, epoch_id, result_dir,
                                      logger)
    elif not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_rcnn(model, dataloader, dataset, epoch_id,
                                       result_dir, logger)
    elif cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_joint(model, dataloader, dataset, epoch_id,
                                        result_dir, logger)
    else:
        raise NotImplementedError
    return ret_dict


def load_ckpt_based_on_args(model, logger):
    """load checkpoint"""
    if args.ckpt is not None:
        train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)
        return

    if cfg.RPN.ENABLED and args.rpn_ckpt is not None:
        # false
        pass

    if cfg.RCNN.ENABLED and args.rcnn_ckpt is not None:
        # false
        pass


def eval_single_ckpt(root_result_dir_):
    """eval single ckpt"""
    root_result_dir_ = os.path.join(root_result_dir_, 'eval')
    # set epoch_id and output dir
    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list else 'no_number'
    root_result_dir_ = os.path.join(root_result_dir_, 'epoch_%s' % epoch_id,
                                    cfg.TEST.SPLIT)
    if args.test:
        root_result_dir_ = os.path.join(root_result_dir_, 'test_mode')

    if args.extra_tag != 'default':
        root_result_dir_ = os.path.join(root_result_dir_, args.extra_tag)
    os.makedirs(root_result_dir_, exist_ok=True)

    log_file = os.path.join(root_result_dir_, 'log_eval_one.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger=logger)

    # create dataloader & network
    test_loader, test_dataset, num_class = create_dataloader(logger)
    model = PointRCNN(num_classes=num_class, use_xyz=True, mode='TEST')

    # copy important files to backup
    backup_dir = os.path.join(root_result_dir_, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp *.py %s/' % backup_dir)
    os.system('cp ../lib/net/*.py %s/' % backup_dir)
    os.system('cp ../lib/datasets/kitti_rcnn_dataset.py %s/' % backup_dir)

    # load checkpoint
    load_ckpt_based_on_args(model, logger)

    # start evaluation
    eval_one_epoch(model, test_loader, test_dataset, epoch_id,
                   root_result_dir_, logger)


def get_no_evaluated_ckpt(ckpt_dir_, ckpt_record_file):
    """get no evaluated ckpt"""
    ckpt_list = glob.glob(os.path.join(ckpt_dir_, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [
        float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()
    ]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if not num_list:
            continue

        epoch_id = num_list[-1]
        if float(epoch_id) not in evaluated_ckpt_list and int(
                float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def create_dataloader(logger):
    """create dataloader"""
    mode = 'TEST' if args.test else 'EVAL'
    DATA_PATH = "data"

    # create dataloader
    test_set = KittiRCNNDataset(
        root_dir=DATA_PATH,
        npoints=cfg.RPN.NUM_POINTS,
        split=cfg.TEST.SPLIT,
        mode=mode,
        random_select=args.random_select,
        rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
        rcnn_eval_feature_dir=args.rcnn_eval_feature_dir,
        classes=cfg.CLASSES,
        logger=logger)
    cols = test_set.getitem_cols(0)
    test_loader = ms.dataset.GeneratorDataset(test_set,
                                              num_parallel_workers=1,
                                              column_names=cols,
                                              shuffle=False)
    test_batch_loader = test_loader.batch(args.batch_size,
                                          drop_remainder=True,
                                          num_parallel_workers=4,
                                          per_batch_map=batchpad(cols=cols),
                                          python_multiprocessing=True)

    return test_batch_loader, test_set, test_set.num_class


if __name__ == "__main__":
    # merge config and log to file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.eval_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('output', 'rpn', cfg.TAG)
        ckpt_dir = os.path.join('output', 'rpn', cfg.TAG, 'ckpt')
    elif args.eval_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join('output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('output', 'rcnn', cfg.TAG, 'ckpt')
    elif args.eval_mode == 'rcnn_offline':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        root_result_dir = os.path.join('output', 'rcnn', cfg.TAG)
        ckpt_dir = os.path.join('output', 'rcnn', cfg.TAG, 'ckpt')
        assert args.rcnn_eval_roi_dir is not None and args.rcnn_eval_feature_dir is not None
    else:
        raise NotImplementedError

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir

    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok=True)

    if args.eval_all:
        assert os.path.exists(ckpt_dir), '%s' % ckpt_dir
    else:
        eval_single_ckpt(root_result_dir)
