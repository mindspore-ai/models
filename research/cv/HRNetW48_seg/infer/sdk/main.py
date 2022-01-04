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
"""The main script for SDK inference."""
import os
import time
import argparse
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
                             MxProtobufIn, StringVector

from cityscapes import Cityscapes


colors = [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32)
    ]


def parse_args():
    """Set and check parameters."""
    parser = argparse.ArgumentParser(description="protonet process")
    parser.add_argument("--pipeline", type=str, default="", help="SDK infer pipeline")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_lst", type=str, default="val.lst")
    parser.add_argument("--label_dir", type=str, default="")
    parser.add_argument("--infer_result_path", type=str, default="")
    args_opt = parser.parse_args()
    return args_opt


def post_process(output, save_path):
    """
    The reasoning results are transformed into gray images,
    and finally the semantic segmentation results are stored
    in the form of color images.

    Args:
        output: Infer result(numpy array) with shape of (C, H, W).
        save_path: Storage path of result image. Example: './result0.png'
    """
    output = output.transpose(1, 2, 0)
    gray = np.argmax(output, axis=2)
    mat = []
    for i in range(1024):
        row = []
        for j in range(2048):
            index = gray[i][j]
            row.append(colors[index])
        mat.append(row)
    image = np.array(mat, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, image)


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """
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


def get_confusion_matrix(label, output, shape, num_classes, ignore_label):
    """Calcute the confusion matrix by given label and pred."""
    output = output.transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(label[:, :shape[-2], :shape[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore_label
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_classes + seg_pred).astype(np.int32)
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred in range(num_classes):
            cur_index = i_label * num_classes + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def main():
    """Read pipeline and do infer."""
    args = parse_args()

    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return

    # create streams by pipeline config file
    with open(os.path.realpath(args.pipeline), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        return

    stream_name = b'segmentation'
    infer_total_time = 0
    loader = Cityscapes(args.data_path, args.data_lst)
    num_classes = 19
    ignore_label = 255
    confusion_matrix = np.zeros((num_classes, num_classes))
    for img, msk, name in loader:
        if not send_source_data(0, img, stream_name, stream_manager_api):
            return
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        infer_total_time += time.time() - start_time
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % infer_result[0].errorCode)
            return
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        # adjust data format
        output = res.reshape((1, 19, 1024, 2048))
        confusion_matrix += get_confusion_matrix(msk, output, [1, 1024, 2048], num_classes, ignore_label)
        # save infer result as image
        result_name = os.path.join(args.infer_result_path, name.replace("leftImg8bit", "result"))
        post_process(output[0], result_name)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    iou_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_iou = iou_array.mean()
    # Show results
    print("========= SDK Inference Result =========")
    print("The total time of inference is {} s".format(infer_total_time))
    print("mIoU:", mean_iou)
    print("IoU array: \n", iou_array)
    print("========================================")

    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    main()
