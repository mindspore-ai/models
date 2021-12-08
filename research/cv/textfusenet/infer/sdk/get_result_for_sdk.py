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
"""evaluation textfusenet"""
import argparse
import json
import os
import cv2
import numpy as np


def parser_args():
    """parse the input"""
    parser = argparse.ArgumentParser(description="textfusenet eval")
    parser.add_argument("--img_path", type=str, required=True, help="image directory")
    parser.add_argument("--infer_result_dir", type=str, required=True, help='cache dir of inference result')
    parser.add_argument("--result_path", type=str, required=True, help="result path")
    args_ = parser.parse_args()
    return args_

def compute_area(poly):
    """compute the area of poly"""
    point = []
    for i in range(0, len(poly) - 1, 2):
        point.append([poly[i], poly[i + 1]])
    s = 0.0
    point_num = len(point)
    if point_num < 3:
        return 0.0
    for i in range(len(point)):
        s += point[i][1] * (point[i-1][0]-point[(i+1)%point_num][0])
    return abs(s/2.0)


def convert_infer(img_path, json_path, result_path):
    """convert infer to result"""
    json_list = os.listdir(json_path)
    for _ in json_list:
        if not _.endswith('.json'):
            continue
        img = cv2.imread(img_path+_.replace('.json', '.jpg'))
        h, w = img.shape[:2]
        data = json.load(open(json_path+_))
        f = open(result_path+'/'+_.replace('.json', '.txt'), 'w')
        if data is not None:
            det_data = data['MxpiObject']
            for bbox in det_data:
                im_mask = np.zeros((h, w), dtype=np.uint8)
                class_vec = bbox.get('classVec')[0]
                if int(class_vec['classId']) > 0 or float(class_vec["confidence"]) < 0.9:
                    continue
                np_bbox = np.array([
                    float(bbox["x0"]),
                    float(bbox["y0"]),
                    float(bbox["x1"]),
                    float(bbox["y1"]),
                    float(class_vec["confidence"])
                ])

                mask_data = bbox['imageMask']['data']
                mask_width = int(bbox['imageMask']['shape'][1])
                mask_height = int(bbox['imageMask']['shape'][0])
                mask = [255 if i == '1' else 0 for i in mask_data]
                mask = np.array(mask)
                mask = mask.reshape(mask_height, mask_width)
                mask = mask.astype(np.uint8)
                [x1, y1, x2, y2] = [int(np_bbox[i]) for i in range(len(np_bbox)-1)]
                roi_w, roi_h = x2 - x1 + 1, y2 - y1 + 1
                if x1 < 0 or y1 < 0 or x1+roi_w > w or y1+roi_h > h:
                    continue
                mask = cv2.resize(mask, (roi_w, roi_h))
                im_mask[y1:y1+roi_h, x1:x1+roi_w] = mask
                temp = cv2.findContours(im_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
                temp = [x.flatten() for x in temp]
                temp = [x for x in temp if len(x) > 6]
                poly = temp
                poly = poly[0]
                area = compute_area(poly)
                if area < 115:
                    continue

                for p in range(0, len(poly) - 1):
                    f.write(str(poly[p]) + ',')
                f.write(str(poly[len(poly) - 1]) + '\n')
        f.close()


if __name__ == "__main__":
    args = parser_args()
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    convert_infer(args.img_path, args.infer_result_dir, args.result_path)
