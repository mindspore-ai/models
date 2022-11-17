#!/usr/bin/env python
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
# limitations under the License.mitations under the License.

import os
import argparse
import mxpiOpenposeProto_pb2 as mxpiOpenposeProto
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import cv2


COCO_PAIRS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
              (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)]  # = 19

COCO_PAIRS_RENDER = COCO_PAIRS[:-2]

COCO_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def gen_parser_args():
    parser = argparse.ArgumentParser(description="maskrcnn inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image directory.")
    parser.add_argument("--res_path",
                        type=str,
                        required=True,
                        help="save directory.")

    res_args = parser.parse_args()
    return res_args


def draw_pose_bbox(npimg, person_list):
    """
    draw person keypoints and skeletons on input image

    Args:
        npimg: input image
        person_list: MxpiPersonList object, each element of which is a MxpiPersonInfo object that stores data of person

    Returns:
        None

    """
    for person_t in person_list:
        skeletons_t = person_t.skeletonInfoVec
        centers = {}
        seen_idx_t = []
        # draw keypoints
        for skele_t in skeletons_t:
            part_idx1_t = skele_t.cocoSkeletonIndex1
            part_idx2_t = skele_t.cocoSkeletonIndex2
            if part_idx1_t not in seen_idx_t:
                seen_idx_t.append(part_idx1_t)
                center = (int(skele_t.x0), int(skele_t.y0))
                centers[part_idx1_t] = center
                cv2.circle(npimg, center, 3, COCO_COLORS[part_idx1_t], -1)

            if part_idx2_t not in seen_idx_t:
                seen_idx_t.append(part_idx2_t)
                center = (int(skele_t.x1), int(skele_t.y1))
                centers[part_idx2_t] = center
                cv2.circle(npimg, center, 3, COCO_COLORS[part_idx2_t], -1)
        # draw skeletons_t
        for pair_order, pair in enumerate(COCO_PAIRS_RENDER):
            if pair[0] not in seen_idx_t or pair[1] not in seen_idx_t:
                continue
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], COCO_COLORS[pair_order], 2)
    return npimg


if __name__ == '__main__':
    args = gen_parser_args()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("Openpose.pipeline", "rb") as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    stream_name = b"classification+detection"
    in_plugin_id = 0
    data_input = MxDataInput()
    file_name = args.img_path
    res_dir = args.res_path
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if os.path.exists(file_name) != 1:
        print("The test image does not exist.")
    with open(file_name, 'rb') as f:
        data_input.data = f.read()
    unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()
    keys = [b"mxpi_openposepostprocess0"]
    key_vec = StringVector()
    for key in keys:
        key_vec.push_back(key)
    infer_result = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, key_vec)
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("infer_result error. errorCode=%d" % (infer_result[0].errorCode))
        exit()
    print("GetProtobuf errorCode=%d" % (infer_result[0].errorCode))
    print("KEY: {}".format(str(infer_result[0].messageName)))
    result_personlist = mxpiOpenposeProto.MxpiPersonList()
    result_personlist.ParseFromString(infer_result[0].messageBuf)
    detect_person_list = result_personlist.personInfoVec
    coco_keypoints = []
    scores = []
    coor_bias = 0.5
    for person in detect_person_list:
        skeletons = person.skeletonInfoVec
        person_score = person.score - 1 # -1 for 'neck'
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        seen_idx = [1]
        # draw keypoints
        for skele in skeletons:
            part_idx1 = skele.cocoSkeletonIndex1
            part_idx2 = skele.cocoSkeletonIndex2 # two end points of a skeleton
            if part_idx1 not in seen_idx:
                seen_idx.append(part_idx1)
                center_x = skele.x0 + coor_bias
                center_y = skele.y0 + coor_bias
                keypoints[to_coco_map[part_idx1] * 3 + 0] = round(center_x, 2)
                keypoints[to_coco_map[part_idx1] * 3 + 1] = round(center_y, 2)
                keypoints[to_coco_map[part_idx1] * 3 + 2] = 1

            if part_idx2 not in seen_idx:
                seen_idx.append(part_idx2)
                center_x = skele.x1 + coor_bias
                center_y = skele.y1 + coor_bias
                keypoints[to_coco_map[part_idx2] * 3 + 0] = round(center_x, 2)
                keypoints[to_coco_map[part_idx2] * 3 + 1] = round(center_y, 2)
                keypoints[to_coco_map[part_idx2] * 3 + 2] = 1
        coco_keypoints.append(keypoints)
        scores.append(person_score)
        with open(os.path.join(res_dir, '{}_1.txt'.format(file_name[-16:-4])), 'a') as f_write:
            f_write.write("keypoints: ")
            f_write.writelines(str(keypoints))
            f_write.write('\n')
            f_write.write("person_score: ")
            f_write.writelines(str(person_score))
            f_write.write('\n')

    img = cv2.imread(file_name)
    image_show = draw_pose_bbox(img, detect_person_list)
    res_file_name = os.path.join(res_dir, '{}_detect_result.jpg'.format(file_name[-16:-4]))
    cv2.imwrite(res_file_name, image_show)

    # destroy streams
    stream_manager_api.DestroyAllStreams()
