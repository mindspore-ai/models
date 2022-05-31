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

import cv2
import numpy as np
from PIL import Image

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector

image_size = (352, 352)

def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="RAS process")
    parser.add_argument("--pipeline", type=str, default="../data/config/ras.pipeline", help="SDK infer pipeline")
    parser.add_argument("--dataset_root_path", type=str, default="../data/data/", help="root path of images")
    parser.add_argument("--save_path", type=str, default="./results/", help="save path of images")
    args_opt = parser.parse_args()
    return args_opt

def normalize(image, mean, std):
    '''

    Args:
        image: a numpy of image
        mean: a list
        std: a list

    Returns:
        img: normalized image
        img_org: original image

    '''
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    return image

def get_data(filename):
    """

    Args:
        filename: a path of data

    Returns:
        a array, shape = (3,image_height,image_width)

    """
    with open(filename, 'rb') as f:
        img = Image.open(f)
        img = img.convert("RGB")
    img = np.array(img)
    img_org = img
    img = img.astype(np.float32)
    img = img / 255.0
    img = cv2.resize(img, image_size)
    mean_ = [0.485, 0.456, 0.406]
    std_ = [0.229, 0.224, 0.225]

    img = normalize(image=img, mean=mean_, std=std_)
    img = normalize(image=img, mean=mean_, std=std_).transpose(2, 0, 1)

    return img, img_org


def get_data_dirlist(data_path):
    ''' get sorted data list'''
    data_dirlist = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        data_dirlist.append(file_path)
    data_dirlist_sorted = sorted(data_dirlist)
    return data_dirlist_sorted

def dataloader(data_path):
    ''' data loader'''
    data = []
    data_orgs = []
    data_dirlist_sorted = get_data_dirlist(data_path)
    for file_path in data_dirlist_sorted:
        data_, data_org = get_data(file_path)
        data.append(data_)
        data_orgs.append(data_org)
    return data, data_orgs

def image_loader(imagename):
    '''read grayscale image'''
    image = Image.open(imagename).convert("L")
    return np.array(image)

def Fmeasure(predict_, groundtruth):
    """

    Args:
        predict: predict image
        gt: ground truth

    Returns:
        Calculate F-measure
    """
    sumLabel = 2 * np.mean(predict_)
    if sumLabel > 1:
        sumLabel = 1
    Label3 = predict_ >= sumLabel
    NumRec = np.sum(Label3)
    LabelAnd = Label3
    gt_t = groundtruth > 0.5
    NumAnd = np.sum(LabelAnd * gt_t)
    num_obj = np.sum(groundtruth)
    if NumAnd == 0:
        p = 0
        r = 0
        FmeasureF = 0
    else:
        p = NumAnd / NumRec
        r = NumAnd / num_obj
        FmeasureF = (1.3 * p * r) / (0.3 * p + r)
    return FmeasureF

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

def main():
    """
    read pipeline and do infer
    """

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

    stream_name = b'ras'
    infer_total_time = 0

    filename = os.path.join(args.dataset_root_path, 'images/')
    gtname = os.path.join(args.dataset_root_path, 'gts/')
    save_path = os.path.realpath(args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    Names = []
    Fs = []
    for data in os.listdir(filename):
        name = data.split('.')[0]
        Names.append(name)
    Names = sorted(Names)
    i = 0
    data_image, data_image_org = dataloader(filename)
    for data, data_org in zip(data_image, data_image_org):
        data_org = np.transpose(data_org, (2, 0, 1))
        data = np.expand_dims(data, 0)
        if not send_source_data(0, data, stream_name, stream_manager_api):
            return
        key_vec = StringVector()
        key_vec.push_back(b'modelInfer')
        start_time = time.time()
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
        infer_total_time += time.time() - start_time
        if infer_result.size() == 0:
            print("inferResult is null")
            return
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
            return
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)

        res = np.resize(res, image_size)
        img = cv2.resize(res, (data_org.shape[2], data_org.shape[1]))
        img = 1.0 / (1.0 + np.exp(np.negative(img)))
        img = img.squeeze()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img * 255
        data_name = Names[i]
        gt = image_loader(os.path.join(gtname, data_name + '.png')) / 255
        fmea = Fmeasure(img / 255, gt)
        print("Fmeasure is %.3f" % fmea)
        save_path_end = os.path.join(save_path, data_name + '.png')
        cv2.imwrite(save_path_end, img)
        print("---------------  %d OK ----------------" % i)
        i += 1
    print("-------------- EVALUATION END --------------------")

    predictpath = save_path

    #calculate F-measure
    gtfiles = sorted([gtname + gt_file for gt_file in os.listdir(gtname)])
    predictfiles = sorted([predictpath + '/' + predictfile for predictfile in os.listdir(predictpath)])

    Fs = []
    for i in range(len(gtfiles)):
        gt = image_loader(gtfiles[i]) / 255
        predict = image_loader(predictfiles[i]) / 255
        fmea = Fmeasure(predict, gt)
        Fs.append(fmea)

    print("Average Fmeasure is %.3f" % np.mean(Fs))

    # destroy streams
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    parse_args()
    main()
