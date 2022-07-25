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
'''
This file gets the jointpoints from image.
'''
from __future__ import division

import os
import json
import numpy as np
import cv2 as cv

from mindspore import Tensor, float32, context
from mindspore import load_checkpoint, load_param_into_net

from yolo.yolo import YOLOV3DarkNet53
from yolo.utils import load_image, detect, statistic_normalize_img
from src.config import config
from src.dataset import flip_pairs
from src.utils.transforms import flip_back, get_affine_transform, bbox2sc
from src.utils.inference import get_final_preds
from src.FastPose import createModel
from src.utils.nms import pose_nms
from src.utils.fn import vis_frame


def detect_bbox():
    '''
    detect_bbox
    '''
    img, ori_img_shape = load_image(config.detect_image, config.yolo_image_size)
    network = YOLOV3DarkNet53(is_training=False)
    param_dict = load_checkpoint(config.yolo_ckpt)
    print('loading model fastpose_ckpt from {}'.format(config.yolo_ckpt))
    load_param_into_net(network, param_dict)
    prediction = network(Tensor(img))
    output_big, output_me, output_small = prediction
    output_big = output_big.asnumpy()
    output_me = output_me.asnumpy()
    output_small = output_small.asnumpy()
    bboxes = detect([output_big, output_me, output_small], ori_img_shape, config.yolo_threshold)
    if config.save_bbox_image:
        image = cv.imread(config.detect_image)
        image = image.astype(np.float32)
        for i in range(bboxes.shape[0]):
            w, h, wi, hi = int(bboxes[i, 0]), int(bboxes[i, 1]),\
                           int(bboxes[i, 2]), int(bboxes[i, 3])
            cv.rectangle(image, (w, h), (w + wi, h + hi), (255, 255, 255), thickness=2)
        name = config.detect_image.split('/')[-1].split('.')[0]
        cv.imwrite(os.path.join(config.result_path, name + "_withBbox.jpg"), image)
    return bboxes

def inference(bboxes):
    '''
    inference
    '''
    image_width = config.MODEL_IMAGE_SIZE[0]
    image_height = config.MODEL_IMAGE_SIZE[1]
    aspect_ratio = image_width * 1.0 / image_height
    scales, centers = bbox2sc(bboxes, aspect_ratio)
    model = createModel()
    ckpt_name = config.fast_pose_ckpt
    print('loading model fastpose_ckpt from {}'.format(ckpt_name))
    load_param_into_net(model, load_checkpoint(ckpt_name))
    image_size = np.array(config.MODEL_IMAGE_SIZE, dtype=np.int32)
    data_numpy = cv.imread(config.detect_image, cv.IMREAD_COLOR | cv.IMREAD_IGNORE_ORIENTATION)




    inputs = []
    bbox_num = bboxes.shape[0]
    image_size = np.array(config.MODEL_IMAGE_SIZE, dtype=np.int32)
    for i in range(bbox_num):
        s, c = scales[i], centers[i]
        trans = get_affine_transform(c, s, 0, image_size, inv=0)
        image = cv.warpAffine(data_numpy, trans, (int(image_size[0]), int(image_size[1])), flags=cv.INTER_LINEAR)
        image_data = np.transpose(statistic_normalize_img(image, statistic_norm=True), (2, 0, 1))
        inputs.append(image_data)
    inputs = np.array(inputs, dtype=np.float32)
    output = model(Tensor(inputs, float32)).asnumpy()
    if config.TEST_FLIP_TEST:
        inputs_flipped = Tensor(inputs[:, :, :, ::-1], float32)
        output_flipped = model(inputs_flipped)
        output_flipped = flip_back(output_flipped.asnumpy(), flip_pairs)

        if config.TEST_SHIFT_HEATMAP:
            output_flipped[:, :, :, 1:] = \
                output_flipped.copy()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5
    preds, maxvals = get_final_preds(output, centers, scales)

    return preds, maxvals

def DataWrite(result):
    '''
    DataWrite
    '''
    img_name = result['imgname'].split('/')[-1].split('.')[0]
    save_path = os.path.join(config.result_path, img_name)

    img = cv.imread(config.detect_image)
    img = vis_frame(img, result)
    cv.imwrite(save_path + ".jpg", img)

    for i in range(len(result['result'])):
        result['result'][i]['keypoints'] = result['result'][i]['keypoints'].tolist()
        result['result'][i]['kp_score'] = result['result'][i]['kp_score'].tolist()
        result['result'][i]['proposal_score'] = result['result'][i]['proposal_score'].tolist()

    with open(save_path + ".json", "w") as f:
        json.dump(result, f)
    print('result was written successfully in {}'.format(config.result_path))

def main():
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.DEVICE_TARGET, save_graphs=False)
    bboxes = detect_bbox()
    pose_preds, pose_scores = inference(bboxes)

    result = pose_nms(bboxes[:, 0:4], bboxes[:, 4], pose_preds, pose_scores)
    result = {'imgname': config.detect_image, 'result': result}
    DataWrite(result)

if __name__ == "__main__":
    main()
