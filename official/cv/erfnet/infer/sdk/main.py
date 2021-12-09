# Copyright (c) 2021. Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
from argparse import ArgumentParser
import cv2
import numpy as np
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn
import torch
from tqdm import tqdm


def resize(img, size, interpolation):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), interpolation)
    return img.resize(size[::-1], interpolation)


def getImgB(img_path_):
    with open(img_path_, 'rb') as f_:
        image = Image.open(f_).convert('RGB')
    image = resize(image, 512, Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    return image.tobytes()


def colormap_cityscapes():
    cmap = np.zeros([20, 3]).astype(np.uint8)
    cmap[0, :] = np.array([128, 64, 128])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([70, 70, 70])
    cmap[3, :] = np.array([102, 102, 156])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])

    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([107, 142, 35])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])

    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])

    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([119, 11, 32])
    cmap[19, :] = np.array([0, 0, 0])
    return cmap


def load_image(fileName):
    return Image.open(fileName)


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.png'])


def is_label(filename):
    return filename.endswith("_labelTrainIds.png")


class iouEval_1:

    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1
        self.reset()

    def reset(self):
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()

    def addBatch(self, x, y):

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        if x.size(1) == 1:
            x_onehot = torch.zeros(
                x.size(0), self.nClasses, x.size(2), x.size(3))
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if y.size(1) == 1:
            y_onehot = torch.zeros(
                y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if self.ignoreIndex != -1:
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0

        tpmult = x_onehot * y_onehot
        tp = torch.sum(
            torch.sum(torch.sum(tpmult, dim=0, keepdim=True),
                      dim=2, keepdim=True),
            dim=3, keepdim=True
        ).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0,
                                           keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot)
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0,
                                           keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou


class cityscapes_val_datapath:

    def __init__(self, root):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        subset = "val"
        self.images_root += subset
        self.labels_root += subset

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in
                          os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in
                            os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        return filename, filenameGt

    def __len__(self):
        return len(self.filenames)


def infer(img_path_, streamManagerApi_):
    dataInput = MxDataInput()

    streamName = b'erfnet'
    dataInput.data = getImgB(img_path_)
    protobufVec = InProtobufVector()
    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()
    visionVec.visionInfo.format = 1
    visionVec.visionData.deviceId = 0
    visionVec.visionData.memType = 0
    visionVec.visionData.dataStr = dataInput.data
    protobuf = MxProtobufIn()
    protobuf.key = b'appsrc0'
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = visionList.SerializeToString()
    protobufVec.push_back(protobuf)
    uniqueId = streamManagerApi_.SendProtobuf(streamName, 0, protobufVec)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    key = b'mxpi_tensorinfer0'
    keyVec = StringVector()
    keyVec.push_back(key)
    inferResult = streamManagerApi_.GetProtobuf(streamName, 0, keyVec)
    if inferResult.size() == 0:
        print("inferResult is null")
        exit()
    if inferResult[0].errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d" % (
            inferResult[0].errorCode))
        exit()
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(inferResult[0].messageBuf)
    vision_data_ = result.tensorPackageVec[0].tensorVec[0].dataStr
    vision_data_ = np.frombuffer(vision_data_, dtype=np.float32)
    shape = result.tensorPackageVec[0].tensorVec[0].tensorShape
    vision_data_ = vision_data_.reshape(shape)
    return vision_data_


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--mode', type=int)
    parser.add_argument('--om_model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    config = parser.parse_args()

    mode = config.mode
    om_model_path = config.om_model_path
    data_path = config.data_path
    output_path = config.output_path

    if mode not in [1, 2]:
        raise RuntimeError(
            "mode must be 1 or 2, which means inferencing data or calculating metric on cityscapes.")

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    pipeline = {
        "erfnet": {
            "stream_config": {
                "deviceId": "0"
            },
            "appsrc0": {
                "props": {
                    "blocksize": "409600"
                },
                "factory": "appsrc",
                "next": "mxpi_tensorinfer0"
            },
            "mxpi_tensorinfer0": {
                "props": {
                    "dataSource": "appsrc0",
                    "modelPath": om_model_path
                },
                "factory": "mxpi_tensorinfer",
                "next": "appsink0"
            },
            "appsink0": {
                "props": {
                    "blocksize": "4096000"
                },
                "factory": "appsink"
            }
        }
    }

    pipelineStr = json.dumps(pipeline).encode()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    if mode == 1:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for img_name in tqdm(os.listdir(data_path)):
            img_path = os.path.join(data_path, img_name)

            vision_data = infer(img_path, streamManagerApi)

            print(vision_data[:, :, 0, 0])
            # exit(0)
            # python3 main.py --mode=1 \
            #     --om_model_path="ERFNet.om" \
            #     --data_path="./data" \
            #     --output_path="./output"

            color = colormap_cityscapes()
            res = np.argmax(vision_data, axis=1)
            res = np.transpose(color[res], (0, 1, 2, 3))
            res = np.squeeze(res, axis=0)

            output_img_path = os.path.join(output_path, img_name)
            cv2.imwrite(output_img_path, res)
    elif mode == 2:
        metrics = iouEval_1(nClasses=20)
        datasets = cityscapes_val_datapath(data_path)
        for image_path, target_path in tqdm(datasets):
            with open(target_path, 'rb') as f:
                target = load_image(f).convert('P')
            target = resize(target, 512, Image.NEAREST)
            target = np.array(target).astype(np.uint32)
            target[target == 255] = 19
            target = target.reshape(512, 1024)
            target = target[np.newaxis, :, :]

            res = infer(image_path, streamManagerApi)
            res = res.reshape(1, 20, 512, 1024)
            preds = torch.Tensor(res.argmax(axis=1).astype(
                np.int32)).unsqueeze(1).long()
            labels = torch.Tensor(target.astype(np.int32)).unsqueeze(1).long()
            metrics.addBatch(preds, labels)

        mean_iou, iou_class = metrics.getIoU()
        mean_iou = mean_iou.item()
        with open("metric.txt", "w") as file:
            print("mean_iou: ", mean_iou, file=file)
            print("iou_class: ", iou_class, file=file)
        print("mean_iou: ", mean_iou)
        print("iou_class: ", iou_class)

    streamManagerApi.DestroyAllStreams()
