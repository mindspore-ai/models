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
Evaluation for RCNN.
"""
import os
import pickle
import time
import xml.etree.ElementTree as ET
from itertools import groupby
from operator import itemgetter

import cv2
import mindspore
import mindspore.dataset
import mindspore.dataset.vision as c_trans
from mindspore import load_param_into_net, load_checkpoint, ops
import numpy as np
from tqdm import tqdm

from src import paths
from src.common.logger import Logger
from src.common.mindspore_utils import MSUtils
from src.model import AlexNetCombine, BBoxNet
from src.utils.util import compute_transto, iou, check_dir, all_cls, label_list
from src.utils.config import config

logger = Logger(os.path.basename(__file__))
mean = [0.453 * 255, 0.433 * 255, 0.398 * 255]
std = [0.270 * 255, 0.268 * 255, 0.279 * 255]
ovthresh = 0.5


def local_norm(img, mean_, std_):
    """

    :param img: image
    :param mean_: mead
    :param std_: std
    :return: normalized image
    """
    B, G, R = cv2.split(img)
    B = (B - mean_[2]) / std_[2]
    G = (G - mean_[1]) / std_[1]
    R = (R - mean_[0]) / std_[0]
    new_img = cv2.merge([R, B, G])

    return new_img

work_nums = config.work_nums
def process_svm(val_data_root=os.path.join(paths.Data.ss_root, 'test'),
                img_data_root=os.path.abspath(paths.Data.jpeg_test),
                svm_thresh_=0.6, batch_size=64, svm_model_path=paths.Model.svm, num_workers=work_nums):
    """

    :param val_data_root: validation data root
    :param img_data_root: image data root
    :param svm_thresh_: svm thresh
    :param batch_size: batch size
    :param svm_model_path: svm model path
    :param num_workers: worker numbers
    :return: svm result
    """

    class FastEvaluatorSVMDataset:
        """
        FastEvaluatorSVMDataset
        """

        def __init__(self, val_data_root, img_data_root):
            self.img_data_root = img_data_root
            logger.info("Start reading test data，ss path：%s，picture path：%s " % (val_data_root, self.img_data_root))
            self.filenames = []
            self.images = {}
            self.rects = []
            pickle_list = os.listdir(val_data_root)
            for pickle_filename in pickle_list:
                val_data_path = os.path.join(val_data_root, pickle_filename)
                with open(val_data_path, 'rb') as f_:
                    val_data = pickle.load(f_)
                    for filename, rects in val_data.items():
                        img_path = os.path.abspath("%s/%s.jpg" % (self.img_data_root, filename))
                        img = cv2.imread(img_path)
                        img = local_norm(img, mean, std)
                        self.images[filename] = img
                        for rect in rects:
                            self.filenames.append(filename)
                            self.rects.append(rect)
            logger.info("Test dataset reading completed，Total number of images: %s，the number of rects: %s " % (
                len(self.images), len(self.filenames)))

        def __len__(self):
            return len(self.filenames)

        def __getitem__(self, idx):
            filename = self.filenames[idx]
            rects = self.rects[idx]
            xmin, ymin, xmax, ymax = rects
            crop_img = self.images[filename][ymin:ymax, xmin:xmax]

            filename = np.array([int(filename)], dtype=np.int32)
            crop_img = np.array(crop_img, dtype=np.float32)
            return filename, rects, crop_img

    svm_result_inner = []

    svm_model = AlexNetCombine(21, phase="test")
    if svm_model_path is not None:
        load_param_into_net(svm_model, load_checkpoint(svm_model_path))
    svm_model.set_train(False)
    ops_softmax = ops.Softmax(axis=1)

    svm_dataset = FastEvaluatorSVMDataset(val_data_root, img_data_root)
    svm_dataloader = mindspore.dataset.GeneratorDataset(svm_dataset, ['filename', 'rects', 'crop_img'], shuffle=False,
                                                        num_parallel_workers=num_workers)
    operations = [
        c_trans.Resize([256, 256]),
        c_trans.CenterCrop((224, 224)),
        c_trans.HWC2CHW()
    ]
    svm_dataloader = svm_dataloader.map(operations=operations, input_columns=["crop_img"])
    svm_dataloader = svm_dataloader.batch(batch_size, drop_remainder=False, num_parallel_workers=num_workers)

    for data in tqdm(svm_dataloader.create_dict_iterator()):
        filenames = data['filename'].asnumpy()
        rects = data['rects'].asnumpy()
        crop_imgs = data['crop_img']
        svm_output = svm_model(crop_imgs)
        cls = svm_output.asnumpy().argmax(axis=1)
        probs = ops_softmax(svm_output).asnumpy()
        bs = probs.shape[0]
        for i in range(bs):
            if probs[i][cls[i]] >= svm_thresh_ and cls[i] != 20:
                svm_result_inner.append(
                    {'filename': filenames[i][0], 'rects': rects[i], 'cls': cls[i],
                     'prob': probs[i][cls[i]]})
    logger.info("svm has finished ,there are %s rects left" % len(svm_result_inner))
    return svm_result_inner


def process_nms(svm_result_inner):
    """

    :param svm_result_inner: svm result
    :return: nms result
    """

    def nms(rect_list, score_list):
        """

        :param rect_list: rect list
        :param score_list: score list
        :return: nms_rects, nms_scores
        """
        nms_rects = list()
        nms_scores = list()

        rect_array = np.array(rect_list)
        score_array = np.array(score_list)

        idxs = np.argsort(score_array)[::-1]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

        thresh = 0.3
        len_score_array = len(score_array)
        while len_score_array > 0:

            nms_rects.append(rect_array[0])
            nms_scores.append(score_array[0])
            rect_array = rect_array[1:]
            score_array = score_array[1:]

            len_score_array = len(score_array)
            if len_score_array <= 0:
                break

            iou_scores = iou(np.array(nms_rects[len(nms_rects) - 1]), rect_array)

            idxs = np.where(iou_scores < thresh)[0]
            rect_array = rect_array[idxs]
            score_array = score_array[idxs]

            len_score_array = len(score_array)
            if len_score_array <= 0:
                break

        return nms_rects, nms_scores

    nms_result_inner = []

    for filename, all_cls_rects in \
            groupby(sorted(svm_result_inner, key=itemgetter('filename')), itemgetter('filename')):
        filename = "%06d" % filename
        score_list, positive_list = [[] for _ in range(20)], [[] for _ in range(20)]
        for line in list(all_cls_rects):
            cls = line['cls']
            score_list[cls].append(line['prob'])
            positive_list[cls].append(line['rects'])
        for i in range(20):
            nms_rects, nms_scores = nms(positive_list[i], score_list[i])
            for ii in range(len(nms_scores)):
                nms_result_inner.append({'filename': filename, 'rects': nms_rects[ii],
                                         'cls': i, 'score': nms_scores[ii]})

    logger.info("nms has finished ,there are %s rects left" % len(nms_result_inner))
    return nms_result_inner


def process_bbox_regression(nms_result_, img_data_root=os.path.abspath(paths.Data.jpeg_test),
                            batch_size=64, reg_model_path=paths.Model.regression, num_workers=work_nums):
    """

    :param nms_result_: nms result
    :param img_data_root: image data root
    :param batch_size: batch size
    :param reg_model_path: regression model path
    :param num_workers: worker numbers
    :return: regression result
    """

    class FastEvaluatorBBoxRegression:
        """
        FastEvaluatorBBoxRegression
        """

        def __init__(self, nms_result_inner, img_data_root):
            self.nms_result = nms_result_inner
            self.images = {}
            for filename, _ in groupby(sorted(nms_result_inner, key=itemgetter('filename')), itemgetter('filename')):
                img_path = os.path.abspath("%s/%s.jpg" % (img_data_root, filename))
                img = cv2.imread(img_path)
                img = local_norm(img, mean, std)
                self.images[filename] = img
            logger.info(
                "Bbox regression data preparation completed,the number of pictures is %s,"
                "and the number of rects is %s" % (len(self.images), len(self.nms_result)))

        def __len__(self):
            return len(self.nms_result)

        def __getitem__(self, idx):
            filename = self.nms_result[idx]['filename']
            rects = self.nms_result[idx]['rects']
            cls = self.nms_result[idx]['cls']
            scores = self.nms_result[idx]['score']

            cls_onehot = np.zeros(20, dtype=np.float32)
            cls_onehot[cls] = 1

            xmin, ymin, xmax, ymax = rects
            img_width = self.images[filename].shape[1]
            img_height = self.images[filename].shape[0]
            crop_img = self.images[filename][ymin:ymax, xmin:xmax]

            filename = np.array([int(filename)], dtype=np.int32)
            img_width = np.array(img_width, dtype=np.int32)
            img_height = np.array(img_height, dtype=np.int32)
            crop_img = np.array(crop_img, dtype=np.float32)
            return filename, rects, crop_img, cls, cls_onehot, scores, img_width, img_height

    cls_list = [[] for _ in range(20)]

    reg_model = BBoxNet()
    if reg_model_path is not None:
        load_param_into_net(reg_model, load_checkpoint(reg_model_path))
    reg_model.set_train(False)

    dataset = FastEvaluatorBBoxRegression(nms_result_, img_data_root=img_data_root)
    dataloader = mindspore.dataset.GeneratorDataset(dataset,
                                                    ['filename', 'rects', 'crop_img', 'cls', 'cls_onehot', 'scores',
                                                     'img_width', 'img_height'], shuffle=False,
                                                    num_parallel_workers=num_workers)
    operations = [
        c_trans.Resize([256, 256]),
        c_trans.CenterCrop((224, 224)),
        c_trans.HWC2CHW()
    ]
    dataloader = dataloader.map(operations=operations, input_columns=["crop_img"])
    dataloader = dataloader.batch(batch_size, drop_remainder=False, num_parallel_workers=num_workers)

    for data in tqdm(dataloader.create_dict_iterator()):
        filenames = data['filename'].asnumpy()
        rects = data['rects'].asnumpy()
        crop_imgs = data['crop_img']
        cls = data['cls'].asnumpy()
        cls_onehot = data['cls_onehot']
        scores = data['scores'].asnumpy()
        img_width = data['img_width'].asnumpy()
        img_height = data['img_height'].asnumpy()

        trans = reg_model(crop_imgs, cls_onehot).asnumpy()

        batch_size = trans.shape[0]
        for i in range(batch_size):
            trans_result, flag = compute_transto(list(rects[i]), trans[i], img_height[i], img_width[i])
            if flag:
                line = list()
                line.append("%06d" % filenames[i][0])
                line.append([int(x) for x in trans_result])
                line.append(scores[i])
                cls_list[cls[i]].append(line)

    return cls_list


def save_result(cls_list, dir_name):
    """

    :param cls_list: result list
    :param dir_name: output
    """
    for i in range(20):
        try:
            img_ids_list, rects_list, scores_list = zip(*cls_list[i])
        except ValueError:
            continue
        rects_list = list(rects_list)
        scores_list = list(scores_list)
        img_ids_list = list(img_ids_list)
        check_dir(dir_name)
        cls_txt = all_cls[i] + '.txt'
        file_name = os.path.join(dir_name, cls_txt)

        with open(file_name, 'w') as f_:
            for ii in range(len(rects_list)):
                rect = rects_list[ii]
                one_line = str(img_ids_list[ii]) + ' ' + str(scores_list[ii]) + ' ' + str(rect[0]) + ' ' + str(
                    rect[1]) + ' ' + str(rect[2]) + ' ' + str(rect[3]) + '\n'
                f_.write(one_line)


def parse_rec(filename):
    """

    :param filename: file to parse
    :return: parse result
    """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec_, prec_):
    """
    spilt from calc_ap to pass lizard
    :param rec_: rec
    :param prec_: prec
    :return: ap
    """
    # 11 point metric
    ap_ = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec_ >= t) == 0:
            p = 0
        else:
            p = np.max(prec_[rec_ >= t])
        ap_ = ap_ + p / 11.
    return ap_


def calc_ap_read_one_file(cls, recs):
    """
    spilt from calc_ap to pass lizard
    :param cls: cls
    :param recs: recs
    :return: class_recs, npos
    """
    cls_valist = os.path.join(paths.Data.image_id_list_test_all_class, cls + '_test.txt')
    class_recs = {}
    npos = 0
    for line in open(cls_valist):
        line = line.strip().split(' ')
        if len(line) == 3 and line[2] == '1':
            pic_id = line[0]
            R = [obj for obj in recs[pic_id] if obj['name'] == cls]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[pic_id] = {'bbox': bbox,
                                  'difficult': difficult,
                                  'det': det}
    return class_recs, npos


def calc_ap(result_file_dir, cls, recs):
    """
    calculate ap
    :param result_file_dir: result file dir
    :param cls: cls
    :param recs: recs
    :return: ap
    """
    class_recs, npos = calc_ap_read_one_file(cls, recs)
    detfile = os.path.join(result_file_dir, cls + '.txt')
    try:
        with open(detfile, 'r') as f_:
            lines = f_.readlines()
    except IOError:
        return None

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        try:
            R = class_recs[image_ids[d]]
        except KeyError:
            fp[d] = 1
            continue
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return voc_ap(rec, prec)


def calc_map(result_file_dir):
    """
    calculate map
    :param result_file_dir: result file dir
    :return: map
    """
    annopath = paths.Data.annotation_test
    recs = {}
    for line in open(paths.Data.image_id_list_test):
        pic_id = line.strip()
        recs[pic_id] = parse_rec(os.path.join(annopath, pic_id + '.xml'))
    ap = list()
    for cls in label_list:
        ap.append(calc_ap(result_file_dir, cls, recs))
    ap = np.array(ap)
    mAP_ = np.sum(ap) / len(label_list)
    return mAP_


if __name__ == '__main__':

    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    device_type = config.device_type
    device_id = config.device_id
    eval_batch_size = config.eval_batch_size
    MSUtils.initialize(device=device_type, device_id=config.device_id)
    mindspore.dataset.config.set_prefetch_size(eval_batch_size * 1)

    svm_result = process_svm(svm_thresh_=0, batch_size=eval_batch_size, num_workers=work_nums,
                             svm_model_path=paths.Model.svm)

    svm_threshes = [0.0, 0.3, 0.6]
    for svm_thresh in svm_threshes:
        svm_result_ = [line for line in svm_result if line['prob'] > svm_thresh]
        logger.info("the thresh is %s, svm has finished ,there are %s rects left" % (svm_thresh, len(svm_result_)))
        nms_result = process_nms(svm_result_)
        cls_list_ = process_bbox_regression(nms_result, batch_size=eval_batch_size, num_workers=work_nums,
                                            reg_model_path=paths.Model.regression)
        save_result(cls_list_, dir_name="bbox_regression_thresh%s_" % svm_thresh + current_time)
        mAP = calc_map("bbox_regression_thresh%s_" % svm_thresh + current_time)
        print("svm_thresh: %s, map: %s" % (svm_thresh, mAP))
