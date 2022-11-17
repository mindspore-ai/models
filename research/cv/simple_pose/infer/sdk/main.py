# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import time
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StringVector
from api.infer import SdkApi
from api.dataset import flip_pairs, keypoint_dataset
from src.utils.transform import flip_back, get_affine_transform
from src.evaluate.coco_eval import evaluate
from src.predict import get_final_preds
from src.model_utils.config import config

def parser_args():
    parser = argparse.ArgumentParser(description="simplepose inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=False,
                        default="../data/input",
                        help="image directory.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="sdk_infer_result",
        help=
        "cache dir of inference result. The default is 'infer_result'."
    )

    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--annfilepath", type=str, default="./files/", help="")
    parser.add_argument("--TEST_COCO_BBOX_FILE", type=str,
                        default="./files/COCO_val2017_detections_AP_H_56_person.json", help="")

    res_args = parser.parse_args()
    return res_args

def process_img(img_file, p_c, p_s):
    # Computed from random subset of ImageNet training images
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    trans_input = get_affine_transform(p_c, p_s, 0, [192, 256])

    inp_img = cv2.warpAffine(img, trans_input, (int(192), int(256)),
                             flags=cv2.INTER_LINEAR)
    inp_img = (inp_img.astype(np.float32) / 255 - mean) / std

    eval_image = inp_img.reshape((1,) + inp_img.shape)
    model_img = eval_image.transpose(0, 3, 1, 2)
    return model_img

if __name__ == '__main__':
    args = parser_args()
    config.TEST.BATCH_SIZE = args.batch_size
    config.TEST.COCO_BBOX_FILE = args.TEST_COCO_BBOX_FILE
    config.DATASET.ROOT = args.annfilepath
    valid_dataset = keypoint_dataset(
        config,
        ann_file="./files/annotations/person_keypoints_val2017.json",
        image_path=args.img_path,
        bbox_file=config.TEST.COCO_BBOX_FILE,
        train_mode=False,
    )

    num_samples = len(valid_dataset.db) * config.TEST.BATCH_SIZE
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 2))
    image_id = []
    idx = 0

    # init stream manager
    sdk_api = SdkApi("./config/simple_pose.pipeline")
    if not sdk_api.init():
        exit(-1)
    img_data_plugin_id = 0
    stream_name = b'im_simplepose'

    # Construct the input of the stream

    res_dir_name = args.infer_result_dir
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    start = time.time()
    for item in range(len(valid_dataset)):
    # for item in range(1):
        inputs = valid_dataset.db[item]['image']
        c = valid_dataset.db[item]['center']
        s = valid_dataset.db[item]['scale']
        sc = valid_dataset.db[item]['score']

        img_np = process_img(inputs, c, s)
        sdk_api.send_tensor_input(stream_name,
                                  img_data_plugin_id, "appsrc0",
                                  img_np.tobytes(), img_np.shape, 0)
        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer_result = sdk_api.get_protobuf(stream_name, 0, keyVec)
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        result1 = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr,
                                dtype='float32').reshape((1, 17, 64, 48))
        result = result1.copy()

        if config.TEST.FLIP_TEST:
            img_flip = img_np[:, :, :, ::-1]
            sdk_api.send_tensor_input(stream_name,
                                      img_data_plugin_id, "appsrc0",
                                      img_flip.tobytes(), img_flip.shape, 0)
            keys = [b"mxpi_tensorinfer0"]
            keyVec = StringVector()
            for key in keys:
                keyVec.push_back(key)
            infer_result_flip = sdk_api.get_protobuf(stream_name, 0, keyVec)
            result_flip = MxpiDataType.MxpiTensorPackageList()

            result_flip.ParseFromString(infer_result_flip[0].messageBuf)
            result_flip1 = np.frombuffer(result_flip.tensorPackageVec[0].tensorVec[0].dataStr,
                                         dtype='float32').reshape((1, 17, 64, 48))
            result_flip = result_flip1.copy()
            result_flip = flip_back(result_flip, flip_pairs)
            if config.TEST.SHIFT_HEATMAP:
                result_flip[:, :, :, 1:] = \
                    result_flip.copy()[:, :, :, 0:-1]
            result = (result+result_flip) * 0.5

        c = np.array(c)
        c = c.reshape(1, 2)
        s = np.array(s)
        s = s.reshape(1, 2)
        score = np.zeros((1), dtype=np.float32)
        score[0] = sc
        file_id_tem = valid_dataset.db[item]['id']
        file_id = []
        file_id.append(file_id_tem)

        preds, maxvals = get_final_preds(config, result.copy(), c, s)

        im = cv2.imread(inputs)

        x = [preds[0][0][0], preds[0][1][0], preds[0][2][0], preds[0][3][0],
             preds[0][4][0], preds[0][5][0], preds[0][6][0], preds[0][7][0], preds[0][8][0],
             preds[0][9][0], preds[0][10][0], preds[0][11][0], preds[0][12][0], preds[0][13][0],
             preds[0][14][0], preds[0][15][0], preds[0][16][0],]
        y = [preds[0][0][1], preds[0][1][1], preds[0][2][1], preds[0][3][1],
             preds[0][4][1], preds[0][5][1], preds[0][6][1], preds[0][7][1], preds[0][8][1],
             preds[0][9][1], preds[0][10][1], preds[0][11][1], preds[0][12][1], preds[0][13][1],
             preds[0][14][1], preds[0][15][1], preds[0][16][1],]

        for i in range(17):
            cv2.circle(im, (int(x[i]), int(y[i])), 1, [0, 255, 85], -1)

        res_file_name = os.path.join(res_dir_name, '{}_detect_result.jpg'.format(inputs[-16:-4]))
        cv2.imwrite(res_file_name, im)
        preds_file_name = os.path.join(res_dir_name, '{}_preds_result.txt'.format(inputs[-16:-4]))
        maxvals_file_name = os.path.join(res_dir_name, '{}_maxvals_result.txt'.format(inputs[-16:-4]))
        np.savetxt(preds_file_name, preds[0], fmt='%f', delimiter=',')
        np.savetxt(maxvals_file_name, maxvals[0], fmt='%f', delimiter=',')

        num_images, _ = preds.shape[:2]

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        all_boxes[idx:idx + num_images, 0] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 1] = score
        image_id.extend(file_id)
        idx += num_images
        if idx % 1024 == 0:
            print('{} samples validated in {} seconds'.format(idx, time.time() - start))
            start = time.time()

    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id))
    ann_path = "./files/person_keypoints_val2017.json"
    _, perf_indicator = evaluate(
        config, all_preds[:idx], args.infer_result_dir, all_boxes[:idx], image_id)
    print("AP:", perf_indicator)
    # destroy streams
