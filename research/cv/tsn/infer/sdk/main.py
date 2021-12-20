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
"""run sdk"""
import os
import sys
import datetime
import cv2
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

from util.transforms import GroupOverSample, ToTorchFormatTensor, Stack, GroupNormalize, GroupScale, GroupCenterCrop

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector

class VideoRecord:
    """Video files"""
    def __init__(self, root_path, row):
        self.root_path = root_path
        self._data = row

    @property
    def path(self):
        return os.path.join(self.root_path, self._data[0])

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


def load_image(modality, image_tmpl, directory, idx):
    """read image"""
    image = []
    if modality in ['RGB', 'RGBDiff']:
        cv_image = cv2.imread(os.path.join(directory, image_tmpl.format(idx)))
        image = [Image.fromarray(cv_image[:, :, ::-1])]
    elif modality == 'Flow':
        x_img = Image.open(os.path.join(directory, image_tmpl.format('x', idx))).convert('L')
        y_img = Image.open(os.path.join(directory, image_tmpl.format('y', idx))).convert('L')

        image = [x_img, y_img]
    return image


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.
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
    return ret


def prepare():
    """prepare for infer"""
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))

    # create streams by pipeline config file
    with open("./pipeline/tsn.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))

    return stream_manager_api


def transforms(modality, input_size, scale_size, data_length):
    """image transformer"""
    input_mean = [104, 117, 128]
    input_std = [1]

    length = 3
    flow_prefix = ""
    test_crops = 10
    if modality == 'Flow':
        test_crops = 1
        length = 10
        input_mean = [128]
        flow_prefix = "flow_"
    elif modality == 'RGBDiff':
        length = 18
        input_mean = input_mean * (1 + data_length)
    transform = []
    if test_crops == 1:
        transform.append(GroupScale(scale_size))
        transform.append(GroupCenterCrop(input_size))
    elif test_crops == 10:
        transform.append(GroupOverSample(input_size, scale_size))

    transform.append(Stack(roll=True))
    transform.append(ToTorchFormatTensor(div=False))
    transform.append(GroupNormalize(input_mean, input_std))
    return flow_prefix, length, transform


def inference(image, stream_name, stream_manager_api):
    """tsn infer"""
    image = np.expand_dims(image, axis=0)
    uniqueId = send_source_data(0, np.array(image, dtype=np.float32), stream_name, stream_manager_api)
    if uniqueId < 0:
        print("Failed to send data to stream.")

    # Obtain the inference result by specifying stream_name and uniqueId.
    start_time = datetime.datetime.now()

    keyVec = StringVector()
    keyVec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager_api.GetProtobuf(stream_name, 0, keyVec)

    end_time = datetime.datetime.now()
    print('sdk run time: {}'.format((end_time - start_time).microseconds))

    if infer_result.size() == 0:
        print("inferResult is null")

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))

    # get infer result
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    # convert the inference result to Numpy array
    result = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape(-1)
    return result


def run():
    """
    read pipeline and do infer
    """
    if len(sys.argv) == 6:
        dataset_path = sys.argv[1]
        test_list = sys.argv[2]
        modality = sys.argv[3]
        num_class = int(sys.argv[4])
        res_dir_name = sys.argv[5]
    else:
        print("Please enter Dataset path| Inference result path ")
        exit(1)
    stream_manager_api = prepare()
    stream_name = b'im_tsn'

    if modality == 'RGB':
        data_length = 1
    elif modality in ['Flow', 'RGBDiff']:
        data_length = 5

    input_size = 224
    test_segments = 25
    test_crops = 10
    scale_size = input_size * 256 // 224
    flow_prefix, length, transform = transforms(modality, input_size, scale_size, data_length)

    test_crops = 1 if modality == "Flow" else 10
    # Construct the input of the stream
    video_list = [VideoRecord(dataset_path, x.strip().split(' ')) for x in open(test_list)]

    output = []
    image_tmpl = "img_{:05d}.jpg" if modality in ["RGB", "RGBDiff"] else flow_prefix+"{}_{:05d}.jpg"

    for index in range(len(video_list)):
        record = video_list[index]
        tick = (record.num_frames - data_length + 1) / float(test_segments)
        segment_indices = np.array([int(tick / 2.0 + tick * x) for x in range(test_segments)]) + 1
        images = list()

        for seg_ind in segment_indices:
            p = int(seg_ind)
            for _ in range(data_length):
                seg_imgs = load_image(modality, image_tmpl, record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        for process in transform:
            images = process(images)
        images = images.reshape((-1, length, images.shape[1], images.shape[2]))

        if modality == 'RGBDiff':
            reverse = list(range(data_length, 0, -1))
            input_c = 3
            input_view = images.reshape((-1, test_segments, data_length + 1, input_c,) + images.shape[1:])
            new_data = input_view[:, :, 1:, :, :, :].copy()
            for x in reverse:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            images = new_data
        label = record.label

        res = []
        for image in images:
            result = inference(image, stream_name, stream_manager_api)
            res.append(result)

        res = np.array(res)
        res = res.reshape((test_crops, test_segments, num_class)).mean(axis=0).reshape((test_segments, 1, num_class))
        output.append([res, label])
    video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
    video_labels = [x[1] for x in output]

    cf = confusion_matrix(video_labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    print('Accuracy: {:.01f}%'.format(np.mean(cls_acc) * 100))

    result = np.array(video_pred).reshape((-1, 1))
    np.savetxt(res_dir_name+'predcitions.txt', result.astype(np.int16), fmt='%d')

    # destroy streams
    stream_manager_api.DestroyAllStreams()

if __name__ == '__main__':
    run()
