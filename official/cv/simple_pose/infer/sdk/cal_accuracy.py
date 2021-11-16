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
import numpy as np
from src.evaluate.coco_eval import evaluate
from src.model_utils.config import config
from api.dataset import keypoint_dataset

def parser_args():
    parser = argparse.ArgumentParser(description="simplepose inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=False,
                        default="/home/data/nku_mindx/whx/val2017/",
                        help="image directory.")
    parser.add_argument("--mxbase_path",
                        type=str,
                        required=False,
                        default="../mxbase/infer_results/",
                        help="image directory.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="../data/sdk_infer_result",
        help=
        "cache dir of inference result. The default is '../data/infer_result'."
    )

    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--annfilepath", type=str, default="./files/", help="")
    parser.add_argument("--TEST_COCO_BBOX_FILE", type=str,
                        default="./files/COCO_val2017_detections_AP_H_56_person.json", help="")

    res_args = parser.parse_args()
    return res_args

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

    # Construct the input of the stream

    res_dir_name = args.infer_result_dir
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    start = time.time()
    for item in range(len(valid_dataset)):
    #for item in range(1):
        inputs = valid_dataset.db[item]['image']
        sc = valid_dataset.db[item]['score']
        s = valid_dataset.db[item]['scale']
        s = np.array(s)
        s = s.reshape(1, 2)
        score = np.zeros((1), dtype=np.float32)
        score[0] = sc

        file_id_tem = valid_dataset.db[item]['id']

        file_id = []
        file_id.append(file_id_tem)
        res_file_name = os.path.join(args.mxbase_path, '{}_1.txt'.format(inputs[-16:-4]))
        file_mxbase = open(res_file_name)
        lines = file_mxbase.readlines()
        row_count = 0
        preds = np.zeros((1, 17, 2),
                         dtype=np.float32)
        maxvals = np.zeros((1, 17, 1),
                           dtype=np.float32)

        line = lines[0].strip().split(' ')
        for i in range(17):
            preds[0][i][0] = line[i+1]
            preds[0][i][1] = line[i+2]

        line = lines[1].strip().split(' ')
        for i in range(17):
            maxvals[0][i][0] = line[i+1]

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
    _, perf_indicator = evaluate(
        config, all_preds[:idx], args.infer_result_dir, all_boxes[:idx], image_id)
    print("AP:", perf_indicator)
    # destroy streams
