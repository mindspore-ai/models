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
import argparse
import os
import time
import datetime
import math
from itertools import product
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn

val_origin_size = True
val_save_result = True

def decode_bbox(bbox, priors, var):
    boxes = np.concatenate((
        priors[:, 0:2] + bbox[:, 0:2] * var[0] * priors[:, 2:4],
        priors[:, 2:4] * np.exp(bbox[:, 2:4] * var[1])), axis=1)  # (xc, yc, w, h)
    boxes[:, :2] -= boxes[:, 2:] / 2    # (x0, y0, w, h)
    boxes[:, 2:] += boxes[:, :2]        # (x0, y0, x1, y1)
    return boxes

def prior_box(image_sizes, min_sizes, steps, clip=False):
    """prior box"""
    feature_maps = [
        [math.ceil(image_sizes[0] / step), math.ceil(image_sizes[1] / step)]
        for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_sizes[1]
                s_ky = min_size / image_sizes[0]
                cx = (j + 0.5) * steps[k] / image_sizes[1]
                cy = (i + 0.5) * steps[k] / image_sizes[0]
                anchors += [cx, cy, s_kx, s_ky]

    output = np.asarray(anchors).reshape([-1, 4]).astype(np.float32)

    if clip:
        output = np.clip(output, 0, 1)

    return output


class Timer():
    def __init__(self):
        self.start_time = 0.
        self.diff = 0.

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.diff = time.time() - self.start_time


class DetectionEngine:
    def __init__(self, cfg):
        self.results = {}
        self.nms_thresh = 0.4
        self.conf_thresh = 0.02
        self.iou_thresh = 0.5
        self.var = [0.1, 0.2]
        self.save_prefix = cfg.result_url
        self.gt_dir = cfg.ground_truth

    def _iou(self, a, b):
        """_iou"""
        A = a.shape[0]
        B = b.shape[0]
        max_xy = np.minimum(
            np.broadcast_to(np.expand_dims(a[:, 2:4], 1), [A, B, 2]),
            np.broadcast_to(np.expand_dims(b[:, 2:4], 0), [A, B, 2]))
        min_xy = np.maximum(
            np.broadcast_to(np.expand_dims(a[:, 0:2], 1), [A, B, 2]),
            np.broadcast_to(np.expand_dims(b[:, 0:2], 0), [A, B, 2]))
        inter = np.maximum((max_xy - min_xy + 1), np.zeros_like(max_xy - min_xy))
        inter = inter[:, :, 0] * inter[:, :, 1]

        area_a = np.broadcast_to(
            np.expand_dims(
                (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1),
            np.shape(inter))
        area_b = np.broadcast_to(
            np.expand_dims(
                (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1), 0),
            np.shape(inter))
        union = area_a + area_b - inter
        return inter / union

    def _nms(self, boxes, threshold=0.5):
        """_nms"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indices = np.where(ovr <= threshold)[0]
            order = order[indices + 1]

        return reserved_boxes

    def write_result(self):
        # save result to file.
        import json
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            if not os.path.isdir(self.save_prefix):
                os.makedirs(self.save_prefix)

            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.results, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def detect(self, boxes, confs, resize, scale, image_path, priors):
        if boxes.shape[0] == 0:
            # add to result
            event_name, img_name = image_path.split('/')
            self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                       'bboxes': []}
            return

        boxes = decode_bbox(np.squeeze(boxes, 0), priors, self.var)
        boxes = boxes * scale / resize

        scores = np.squeeze(confs, 0)[:, 1]
        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        # print("boxes: ", boxes)

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._nms(dets, self.nms_thresh)
        dets = dets[keep, :]

        dets[:, 2:4] = (dets[:, 2:4].astype(np.int_) - dets[:, 0:2].astype(np.int_)).astype(np.float_)  # int
        dets[:, 0:4] = dets[:, 0:4].astype(np.int_).astype(np.float_)  # int

        # add to result
        event_name, img_name = image_path.split('/')
        if event_name not in self.results.keys():
            self.results[event_name] = {}
        self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                   'bboxes': dets[:, :5].astype(np.float_).tolist()}

    def _get_gt_boxes(self):
        from scipy.io import loadmat
        gt = loadmat(os.path.join(self.gt_dir, 'wider_face_val.mat'))
        hard = loadmat(os.path.join(self.gt_dir, 'wider_hard_val.mat'))
        medium = loadmat(os.path.join(self.gt_dir, 'wider_medium_val.mat'))
        easy = loadmat(os.path.join(self.gt_dir, 'wider_easy_val.mat'))

        faceboxes = gt['face_bbx_list']
        events = gt['event_list']
        files = gt['file_list']

        hard_gt_list = hard['gt_list']
        medium_gt_list = medium['gt_list']
        easy_gt_list = easy['gt_list']

        return faceboxes, events, files, hard_gt_list, medium_gt_list, easy_gt_list

    def _norm_pre_score(self):
        max_score = 0
        min_score = 1

        for event in self.results:
            for name in self.results[event].keys():
                bbox = np.array(self.results[event][name]['bboxes']).astype(np.float_)
                if bbox.shape[0] <= 0:
                    continue
                max_score = max(max_score, np.max(bbox[:, -1]))
                min_score = min(min_score, np.min(bbox[:, -1]))

        length = max_score - min_score
        for event in self.results:
            for name in self.results[event].keys():
                bbox = np.array(self.results[event][name]['bboxes']).astype(np.float_)
                if bbox.shape[0] <= 0:
                    continue
                bbox[:, -1] -= min_score
                bbox[:, -1] /= length
                self.results[event][name]['bboxes'] = bbox.tolist()

    def _image_eval(self, predict, gt, keep, iou_thresh, section_num):
        _predict = predict.copy()
        _gt = gt.copy()

        image_p_right = np.zeros(_predict.shape[0])
        image_gt_right = np.zeros(_gt.shape[0])
        proposal = np.ones(_predict.shape[0])

        # x1y1wh -> x1y1x2y2
        _predict[:, 2:4] = _predict[:, 0:2] + _predict[:, 2:4]
        _gt[:, 2:4] = _gt[:, 0:2] + _gt[:, 2:4]

        ious = self._iou(_predict[:, 0:4], _gt[:, 0:4])
        for i in range(_predict.shape[0]):
            gt_ious = ious[i, :]
            max_iou, max_index = gt_ious.max(), gt_ious.argmax()
            if max_iou >= iou_thresh:
                if keep[max_index] == 0:
                    image_gt_right[max_index] = -1
                    proposal[i] = -1
                elif image_gt_right[max_index] == 0:
                    image_gt_right[max_index] = 1

            right_index = np.where(image_gt_right == 1)[0]
            image_p_right[i] = len(right_index)

        image_pr = np.zeros((section_num, 2), dtype=np.float_)
        for section in range(section_num):
            _thresh = 1 - (section + 1) / section_num
            over_score_index = np.where(predict[:, 4] >= _thresh)[0]
            if over_score_index.shape[0] <= 0:
                image_pr[section, 0] = 0
                image_pr[section, 1] = 0
            else:
                index = over_score_index[-1]
                p_num = len(np.where(proposal[0:(index + 1)] == 1)[0])
                image_pr[section, 0] = p_num
                image_pr[section, 1] = image_p_right[index]

        return image_pr

    def get_eval_result(self):
        self._norm_pre_score()
        facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = self._get_gt_boxes()
        section_num = 1000
        sets = ['easy', 'medium', 'hard']
        set_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
        ap_key_dict = {0: "Easy   Val AP : ", 1: "Medium Val AP : ", 2: "Hard   Val AP : ",}
        ap_dict = {}
        for _set in range(len(sets)):
            gt_list = set_gts[_set]
            count_gt = 0
            pr_curve = np.zeros((section_num, 2), dtype=np.float_)
            for i, _ in enumerate(event_list):
                event = str(event_list[i][0][0])
                image_list = file_list[i][0]
                event_predict_dict = self.results[event]
                event_gt_index_list = gt_list[i][0]
                event_gt_box_list = facebox_list[i][0]

                for j, _ in enumerate(image_list):
                    predict = np.array(event_predict_dict[str(image_list[j][0][0])]['bboxes']).astype(np.float_)
                    gt_boxes = event_gt_box_list[j][0].astype('float')
                    keep_index = event_gt_index_list[j][0]
                    count_gt += len(keep_index)

                    if gt_boxes.shape[0] <= 0 or predict.shape[0] <= 0:
                        continue
                    keep = np.zeros(gt_boxes.shape[0])
                    if keep_index.shape[0] > 0:
                        keep[keep_index - 1] = 1

                    image_pr = self._image_eval(predict, gt_boxes, keep,
                                                iou_thresh=self.iou_thresh,
                                                section_num=section_num)
                    pr_curve += image_pr

            precision = pr_curve[:, 1] / pr_curve[:, 0]
            recall = pr_curve[:, 1] / count_gt

            precision = np.concatenate((np.array([0.]), precision, np.array([0.])))
            recall = np.concatenate((np.array([0.]), recall, np.array([1.])))

            for i in range(precision.shape[0] - 1, 0, -1):
                precision[i - 1] = np.maximum(precision[i - 1], precision[i])
            index = np.where(recall[1:] != recall[:-1])[0]
            ap = np.sum((recall[index + 1] - recall[index]) * precision[index + 1])

            print(ap_key_dict[_set] + '{:.4f}'.format(ap))

        return ap_dict

def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = array_bytes
    tensor_vec.tensorDataSize = len(array_bytes)
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    return True


def create_val_dataset(args_opt):
    testset_folder = args_opt.data_path
    testset_label_path = args_opt.data_path + "val_label.txt"
    with open(testset_label_path, 'r') as f:
        _test_dataset = f.readlines()
        test_dataset = []
        for im_path in _test_dataset:
            if im_path.startswith('# '):
                test_dataset.append(im_path[2:-1])  # delete '# ...\n'

    num_images = len(test_dataset)

    if val_origin_size:
        h_max, w_max = 0, 0
        for img_name in test_dataset:
            image_path = os.path.join(testset_folder, img_name)
            _img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if _img.shape[0] > h_max:
                h_max = _img.shape[0]
            if _img.shape[1] > w_max:
                w_max = _img.shape[1]

        h_max = 5568
        w_max = 1056

        priors = prior_box(image_sizes=(h_max, w_max),
                           min_sizes=[[16, 32], [64, 128], [256, 512]],
                           steps=[8, 16, 32],
                           clip=False)

    return test_dataset, h_max, w_max, num_images, priors


def val_with_resnet(args_opt):
    if val_origin_size == 0:
        target_size = 1600
        max_size = 2176
        priors = prior_box(image_sizes=(max_size, max_size),
                           min_sizes=[[16, 32], [64, 128], [256, 512]],
                           steps=[8, 16, 32],
                           clip=False)
        # init streams
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(args_opt.PL_PATH, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # init detection engine and timers
    detection = DetectionEngine(args_opt)
    timers = {'forward_time': Timer(), 'misc': Timer()}

    # create testing dataset
    test_dataset, h_max, w_max, num_images, priors = create_val_dataset(args_opt)

    # testing begin
    print('Predict box starting')
    for i, img_name in enumerate(test_dataset):
        stream_name = b'im_Retinaface'
        in_plugin_id = 0
        image_path = os.path.join(args_opt.data_path, img_name)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        if val_origin_size:
            resize = 1
            assert img.shape[0] <= h_max and img.shape[1] <= w_max
            image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
            image_t[:, :] = (104.0, 117.0, 123.0)
            image_t[0:img.shape[0], 0:img.shape[1]] = img
            img = image_t
        else:
            im_size_min = np.min(img.shape[0:2])
            im_size_max = np.max(img.shape[0:2])
            resize = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)

            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

            assert img.shape[0] <= max_size and img.shape[1] <= max_size
            image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
            image_t[:, :] = (104.0, 117.0, 123.0)
            image_t[0:img.shape[0], 0:img.shape[1]] = img
            img = image_t

        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        # get inferring result
        timers['forward_time'].start()
        if not send_source_data(0, img, stream_name, stream_manager_api):
            return
        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer_results = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, keyVec)
        if infer_results.size() == 0 or infer_results.size() == 0:
            print("infer_result is null")
            exit()

        if infer_results[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                infer_results[0].errorCode, infer_results[0].data.decode()))
            exit()
        resultList = MxpiDataType.MxpiTensorPackageList()
        resultList.ParseFromString(infer_results[0].messageBuf)
        boxes = np.frombuffer(resultList.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4').reshape((1, 241164, 4))
        confs = np.frombuffer(resultList.tensorPackageVec[0].tensorVec[1].dataStr, dtype='<f4').reshape((1, 241164, 2))

        print("------image_name is", img_name)
        timers['forward_time'].end()

        # detecting begin
        timers['misc'].start()
        detection.detect(boxes, confs, resize, scale, img_name, priors)
        timers['misc'].end()

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images,
                                                                                     timers['forward_time'].diff,
                                                                                     timers['misc'].diff))
    print('Predict box done.')
    # evaling begin
    print('Eval starting')

    if val_save_result:
        # Save the predict result if you want.
        predict_result_path = detection.write_result()
        print('predict result path is {}'.format(predict_result_path))

    detection.get_eval_result()
    print('Eval done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface Image Inferring sdk')
    # Datasets
    parser.add_argument('--data_path', default='../data/input_main_part/val_images/', type=str,
                        help='data path')
    parser.add_argument('--PL_PATH', default='./config/Retinaface.pipeline', type=str,
                        help='pipeline path')
    parser.add_argument('--result_url', default='./sdk_out/', type=str,
                        help='result path')
    parser.add_argument('--ground_truth', default='../data/input_main_part/ground_truth/', type=str,
                        help='ground truth path')
    args = parser.parse_args()

    val_with_resnet(args_opt=args)
