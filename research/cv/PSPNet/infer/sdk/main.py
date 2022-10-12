# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
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
"""the main sdk infer file"""
import argparse
import base64
import json
import os
import cv2
import numpy as np
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector


def _parse_args():
    parser = argparse.ArgumentParser('mindspore PSPNet eval')
    parser.add_argument('--data_root', type=str, default='',
                        help='root path of val data')
    parser.add_argument('--data_lst', type=str, default='',
                        help='list of val data')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='number of classes')
    parser.add_argument('--result_path', type=str, default='./result',
                        help='the result path')
    parser.add_argument('--color_txt', type=str,
                        default='',
                        help='the color path')
    parser.add_argument('--name_txt', type=str,
                        default='',
                        help='the name_txt path')
    parser.add_argument('--pipeline_path', type=str,
                        default='',
                        help='root path of pipeline file')
    args, _ = parser.parse_known_args()
    return args


def _cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(
        n * a[k].astype(np.int32) + b[k].astype(np.int32), minlength=n ** 2).reshape(n, n)


def _init_stream(pipeline_path):
    """
    initial sdk stream before inference

    Returns:
        stream manager api
    """
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        raise RuntimeError(f"Failed to init stream manager, ret={ret}")

    with open(pipeline_path, 'rb') as f:
        pipeline_str = f.read()

        ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
        if ret != 0:
            raise RuntimeError(f"Failed to create stream, ret={ret}")
        return stream_manager_api


def _do_infer(stream_manager_api, data_input):
    """
    send images into stream to do infer

    Returns:
        infer result, numpy array
    """
    stream_name = b'segmentation'
    unique_id = stream_manager_api.SendDataWithUniqueId(
        stream_name, 0, data_input)
    if unique_id < 0:
        raise RuntimeError("Failed to send data to stream.")
    print("success to send data to stream.")

    timeout = 3000
    infer_result = stream_manager_api.GetResultWithUniqueId(
        stream_name, unique_id, timeout)
    if infer_result.errorCode != 0:
        raise RuntimeError(
            "GetResultWithUniqueId error, errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))

    load_dict = json.loads(infer_result.data.decode())
    image_mask = load_dict["MxpiImageMask"][0]
    data_str = base64.b64decode(image_mask['dataStr'])
    shape = image_mask['shape']
    return np.frombuffer(data_str, dtype=np.uint8).reshape(shape)


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
    print("Success to send data to stream.")
    return True


def get_result(stream_name, stream_manager_api):
    """
    # Obtain the inference result by specifying streamName and uniqueId.
    """
    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
    if infer_result.size() == 0:
        print("inferResult is null")
        return 0
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" %
              (infer_result[0].errorCode))
        return 0
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    vision_data_ = result.tensorPackageVec[0].tensorVec[0].dataStr
    vision_data_ = np.frombuffer(vision_data_, dtype=np.float32)
    shape = result.tensorPackageVec[0].tensorVec[0].tensorShape
    mask_image = vision_data_.reshape(shape)
    return mask_image[0]


def scale_process(stream_manager_api, image, classes, crop_h, crop_w, ori_h, ori_w,
                  mean, std, stride_rate, flip):
    stream_name = b'segmentation'
    print("image=", image.shape)
    ori_h1, ori_w1, _ = image.shape

    pad_h = max(crop_h - ori_h1, 0)
    pad_w = max(crop_w - ori_w1, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape

    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    print("grid_h, grid_w=", grid_h, grid_w)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            print("s_h:e_h, s_w:e_w==", s_h, e_h, s_w, e_w)
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            mean = np.array(mean).astype(np.float32)
            std = np.array(std).astype(np.float32)

            image_crop = image_crop.transpose(2, 0, 1)
            image_crop = (image_crop - mean[:, None, None]) / std[:, None, None]
            image_crop = np.expand_dims(image_crop, 0)
            if not send_source_data(0, image_crop, stream_name, stream_manager_api):
                return 0

            mask_image = get_result(stream_name, stream_manager_api)
            mask_image = mask_image.transpose(1, 2, 0)


            if flip:
                image_crop = np.flip(image_crop, axis=3)
                if not send_source_data(0, image_crop, stream_name, stream_manager_api):
                    return 0
                mask_image_flip = get_result(stream_name, stream_manager_api).transpose(1, 2, 0)
                mask_image_flip = np.flip(mask_image_flip, axis=1)
                mask_image = (mask_image + mask_image_flip) / 2

            prediction_crop[s_h:e_h, s_w:e_w, :] += mask_image

    prediction_crop /= np.expand_dims(count_crop, 2)  # (473, 512, 21)
    print(f"prediction_crop = {pad_h_half}:{pad_h_half + ori_h1},{pad_w_half}:{pad_w_half + ori_w1}")
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h1, pad_w_half:pad_w_half + ori_w1]
    prediction = cv2.resize(prediction_crop, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
    return prediction


def check_makedirs(dir_name):
    """ check file dir """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def colorize(gray, palette):
    """ gray: numpy array of the label and 1*3N size list palette 列表调色板 """
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def intersectionAndUnion(output, target, K, ignore_index=255):
    """
    'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    """

    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    print("output.shape=", output.shape)
    print("output.size=", output.size)
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]

    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def cal_acc(data_root, data_list, pred_folder, classes, names):
    """ Calculation evaluating indicator """
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    with open(data_list) as f:
        img_lst = f.readlines()
        for i, line in enumerate(img_lst):
            image_path, target_path = line.strip().split(' ')
            image_name = image_path.split('/')[-1].split('.')[0]
            pred = cv2.imread(os.path.join(pred_folder, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
            target = cv2.imread(os.path.join(data_root, target_path), cv2.IMREAD_GRAYSCALE)
            intersection, union, target = intersectionAndUnion(pred, target, classes)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            print('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(
                i + 1, len(data_list), image_name + '.png', accuracy))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(classes):
            print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(
                i, iou_class[i], accuracy_class[i], names[i]))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        """ calculate the result """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = _parse_args()

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    crop_h = 473
    crop_w = 473
    classes = 21
    long_size = 512
    gray_folder = os.path.join(args.result_path, 'gray')
    color_folder = os.path.join(args.result_path, 'color')
    colors = np.loadtxt(args.color_txt).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.name_txt)]
    stream_manager_api = _init_stream(args.pipeline_path)
    if not stream_manager_api:
        exit(1)

    with open(args.data_lst) as f:
        img_lst = f.readlines()

        os.makedirs(args.result_path, exist_ok=True)
        for _, line in enumerate(img_lst):
            img_path, msk_path = line.strip().split(' ')
            print("--------------------------------------------")

            img_path = os.path.join(args.data_root, img_path)
            print("img_path:", img_path)
            print("msk_paty:", msk_path)

            ori_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            ori_image = np.float32(ori_image)

            ori_h, ori_w, _ = ori_image.shape
            print("ori_h=", ori_h)
            print("ori_w=", ori_w)
            new_h = long_size
            new_w = long_size
            if ori_h > ori_w:
                new_w = round(long_size / float(ori_h) * ori_w)
            else:
                new_h = round(long_size / float(ori_w) * ori_h)
            print(f"new_w, new_h = ({new_w}, {new_h})")
            image = cv2.resize(ori_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction = scale_process(stream_manager_api, image, classes, crop_h, crop_w, ori_h, ori_w, mean, std=std,
                                       stride_rate=2 / 3, flip=True)
            print("prediction0.shape=", prediction.shape)

            prediction = np.argmax(prediction, axis=2)

            check_makedirs(gray_folder)
            check_makedirs(color_folder)
            gray = np.uint8(prediction)
            color = colorize(gray, colors)
            image_name = img_path.split('/')[-1].split('.')[0]
            gray_path = os.path.join(gray_folder, image_name + '.png')
            color_path = os.path.join(color_folder, image_name + '.png')
            cv2.imwrite(gray_path, gray)
            color.save(color_path)
    stream_manager_api.DestroyAllStreams()

    cal_acc(args.data_root, args.data_lst, gray_folder, classes, names)


if __name__ == '__main__':
    main()
