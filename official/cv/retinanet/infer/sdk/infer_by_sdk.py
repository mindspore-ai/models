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
"""infer_by_sdk"""
import argparse
import json
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from StreamManagerApi import MxDataInput, StringVector
from StreamManagerApi import StreamManagerApi
import MxpiDataType_pb2 as MxpiDataType

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='"Retinanet infer " "example."')

NUM_CLASSES = 81
MIN_SCORE = 0.1
MAX_BOXES = 100
NMS_THERSHOLD = 0.6

# Gets the Full Path to the Current Script
current_path = os.path.abspath(os.path.dirname(__file__))

# pipeline directory
parser.add_argument(
    "--pipeline_path",
    type=str,
    help="mxManufacture pipeline file path",
    default=os.path.join(current_path, "conf/retinanet.pipeline"),
)
parser.add_argument(
    "--stream_name",
    type=str,
    help="Infer stream name in the pipeline config file",
    default="retinanet",
)
parser.add_argument(
    "--img_path",
    type=str,
    help="Image pathname, can be a image file or image directory",
    default=os.path.join(current_path, "dataset/val2017/"),
)
parser.add_argument(
    "--instances_path",
    type=str,
    help="The annotation file directory for the COCO dataset",
    default=os.path.join(current_path, "dataset/annotations/instances_val2017.json"),
)
parser.add_argument(
    "--label_path",
    type=str,
    help="Coco label directory",
    default=os.path.join(current_path, "conf/coco.names"),
)
parser.add_argument(
    "--res_path",
    type=str,
    help="Directory to store the inferred result",
    default=os.path.join(current_path, "result/"),
    required=False,
)

# Analytical Parameters
args = parser.parse_args()

def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep

def send_data_get_output(stream_name, data_input, stream_manager):
    # plug-in id
    in_plugin_id = 0

    # Send data to the plug-in
    unique_id = stream_manager.SendData(stream_name, in_plugin_id, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()
    plugin_names = [b"mxpi_tensorinfer0"]
    name_vector = StringVector()
    for name in plugin_names:
        name_vector.push_back(name)

    infer_result = stream_manager.GetProtobuf(stream_name, 0, name_vector)
    if infer_result[0].errorCode != 0:
        error_message = "GetProtobuf error. errorCode=%d, errorMessage=%s" % (
            infer_result[0].errorCode, infer_result[0].messageName)
        raise AssertionError(error_message)
    # 将infer_result.data即元器件返回的结果转为dict格式，用get("MxpiObject", [])新建一个MxpiObject的Key且复制为[]
    tensor_package = MxpiDataType.MxpiTensorPackageList()
    tensor_package.ParseFromString(infer_result[0].messageBuf)
    box = np.frombuffer(tensor_package.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4').reshape(67995,
                                                                                                      4)
    score = np.frombuffer(tensor_package.tensorPackageVec[0].tensorVec[1].dataStr, dtype='<f4').reshape(67995,
                                                                                                        81)
    return box, score

def parse_img_infer_result(sample, predictions, val_cls_dict, classs_dict):
    pred_boxes = sample['boxes']
    box_scores = sample['box_scores']
    img_id_card = sample['img_id']
    img_size = sample['image_shape']
    h, w = img_size[0], img_size[1]

    final_boxes = []
    final_label = []
    final_score = []

    for c in range(1, NUM_CLASSES):
        class_box_scores = box_scores[:, c]
        score_mask = class_box_scores > MIN_SCORE
        class_box_scores = class_box_scores[score_mask]
        class_boxes = pred_boxes[score_mask] * [h, w, h, w]
        if score_mask.any():
            nms_index = apply_nms(class_boxes, class_box_scores, NMS_THERSHOLD, MAX_BOXES)
            class_boxes = class_boxes[nms_index]
            class_box_scores = class_box_scores[nms_index]
            final_boxes += class_boxes.tolist()
            final_score += class_box_scores.tolist()
            final_label += [classs_dict[val_cls_dict[c]]] * len(class_box_scores)

    for loc, label, score in zip(final_boxes, final_label, final_score):
        res = {}
        res['image_id'] = img_id_card
        res['bbox'] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]]
        res['score'] = score
        res['category_id'] = label
        predictions.append(res)
    print("Parse the box success")

# Images reasoning
def infer():
    """Infer images by Retinanet.    """

    # Create StreamManagerApi object
    stream_manager_api = StreamManagerApi()
    # Use InitManager method init StreamManagerApi
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(args.pipeline_path, "rb") as f:
        pipeline_str = f.read()

    # Configuring a stream
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()
    # Stream_name encoded in UTF-8
    stream_name = args.stream_name.encode()
    print(stream_name)
    predictions = []
    with open(args.label_path, 'rt') as f:
        val_cls = f.read().rstrip("\n").split("\n")
    val_cls_dict = {}
    for i, cls in enumerate(val_cls):
        val_cls_dict[i] = cls
    coco_gt = COCO(args.instances_path)
    classs_dict = {}
    cat_ids = coco_gt.loadCats(coco_gt.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["name"]] = cat["id"]

    for file_name in os.listdir(args.img_path):
        pred_data = []
        # Gets the Address of each image
        img_id = int(file_name.split('.')[0])
        file_path = args.img_path + file_name
        size = (cv2.imread(file_path)).shape

        # Read each photo in turn
        with open(file_path, "rb") as f:
            img_data = f.read()
            if not img_data:
                print(f"read empty data from img:{file_name}")
                continue
        # The element value img_data
        data_input.data = img_data
        boxes_output, scores_output = send_data_get_output(stream_name, data_input, stream_manager_api)
        pred_data.append({"boxes": boxes_output,
                          "box_scores": scores_output,
                          "img_id": img_id,
                          "image_shape": size})

        parse_img_infer_result(pred_data[0], predictions, val_cls_dict, classs_dict)
        print(f"Inferred image:{file_name} success!")

    # Save the result in JSON format
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    with open(args.res_path + 'predictions_test.json', 'w') as f:
        json.dump(predictions, f)
    stream_manager_api.DestroyAllStreams()


if __name__ == "__main__":
    args = parser.parse_args()
    infer()
