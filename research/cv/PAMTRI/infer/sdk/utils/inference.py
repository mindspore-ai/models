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
''' infer module '''
import datetime
import os
import numpy as np
import MxpiDataType_pb2 as MxpiDataType

from utils.transforms import image_proc, gene_mt_input, flip_back, flip_pairs, get_posenet_preds
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn


class SdkInfer:
    ''' sdk infer '''
    def __init__(self, pipeline_path):
        self.pipline_path = pipeline_path
        self._stream_api = None
        self._data_input = None
        self._device_id = None
        self.stream_name = None

    def init_stream(self):
        ''' init stream '''
        stream_manager_api = StreamManagerApi()
        ret = stream_manager_api.InitManager()
        if ret != 0:
            print(f"Failed to init stream manager, ret={ret}.")
            return False
        # with open('../pipline/posenet.pipline', 'rb') as pl:
        with open(self.pipline_path, 'rb') as pl:
            pipline_stream = pl.read()
        ret = stream_manager_api.CreateMultipleStreams(pipline_stream)
        if ret != 0:
            print(f"Failed to create stream, ret={ret}.")
            return False
        self._stream_api = stream_manager_api
        return True

    def send_vision_buf(self, stream_name, data, in_plug_id):
        ''' send visionbuf to model '''
        if self._stream_api is None:
            print('stream_api is None')
            return False
        input_data = data.tobytes()
        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionInfo.format = 13
        vision_vec.visionInfo.width = 256
        vision_vec.visionInfo.height = 256
        vision_vec.visionInfo.widthAligned = 256
        vision_vec.visionInfo.heightAligned = 256
        vision_vec.visionData.memType = 0
        vision_vec.visionData.dataStr = input_data
        vision_vec.visionData.dataSize = len(input_data)

        key = "appsrc{}".format(0).encode('utf-8')
        protobuf_vec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b"MxTools.MxpiVisionList"
        protobuf.protobuf = vision_list.SerializeToString()
        protobuf_vec.push_back(protobuf)
        unique_id = self._stream_api.SendProtobuf(
            stream_name, in_plug_id, protobuf_vec)
        if unique_id < 0:
            print("Failed to send data to stream.")
            return False
        return unique_id

    def send_package_buf(self, stream_name, data, appsrc_pos):
        ''' send packagebuf to model '''
        # create MxpiTensorPackageList
        tensor_package_list = MxpiDataType.MxpiTensorPackageList()
        tensor_package = tensor_package_list.tensorPackageVec.add()
        if isinstance(data, list):
            data = np.array(data)
        array_bytes = data.tobytes()
        data_input = MxDataInput()
        data_input.data = array_bytes
        tensor_vec = tensor_package.tensorVec.add()
        tensor_vec.deviceId = 0
        tensor_vec.memType = 0
        for i in data.shape:
            tensor_vec.tensorShape.append(i)
        tensor_vec.dataStr = data_input.data
        tensor_vec.tensorDataSize = len(array_bytes)

        key = "appsrc{}".format(appsrc_pos).encode('utf-8')
        protobuf_vec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b"MxTools.MxpiTensorPackageList"
        protobuf.protobuf = tensor_package_list.SerializeToString()
        protobuf_vec.push_back(protobuf)
        unique_id = self._stream_api.SendProtobuf(
            stream_name, appsrc_pos, protobuf_vec)
        return unique_id

    def send_mxdata(self, stream_name, data, appsrc_pos):
        ''' send_mxdata '''
        data_input = MxDataInput()
        data_input.data = data.tobytes()
        unique_id = self._stream_api.SendData(
            stream_name, appsrc_pos, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            return False
        return True

    def get_result(self, stream_name, appsrc_pos=0):
        ''' get result bytes and convert to array '''
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        infer_result = self._stream_api.GetProtobuf(
            stream_name, appsrc_pos, key_vec)
        if infer_result[0].errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            return False
        if stream_name == b'PoseEstNet0':
            return self._convert_posenet_result(infer_result)
        if stream_name == b'MultiTaskNet0':
            return self._convert_multitask_result(infer_result)
        return None

    def destroy(self):
        ''' destroy stream '''
        self._stream_api.DestroyAllStreams()

    @staticmethod
    def _convert_posenet_result(infer_result):
        ''' convert bytes to array '''
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        tensor_vec = result.tensorPackageVec[0].tensorVec[0]
        data_str = tensor_vec.dataStr
        tensor_shape = tensor_vec.tensorShape
        infer_array = np.frombuffer(data_str, dtype=np.float32)
        infer_array.shape = tensor_shape
        return infer_array

    @staticmethod
    def _convert_multitask_result(infer_result):
        ''' convert bytes to array '''
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        package_vec = result.tensorPackageVec[0]

        tensor_vec_ids = package_vec.tensorVec[0]
        data_str_ids = tensor_vec_ids.dataStr
        tensor_shape_ids = tensor_vec_ids.tensorShape
        infer_array_ids = np.frombuffer(data_str_ids, dtype=np.float32)
        infer_array_ids.shape = tensor_shape_ids

        tensor_vec_cls = package_vec.tensorVec[1]
        data_str_cls = tensor_vec_cls.dataStr
        tensor_shape_cls = tensor_vec_cls.tensorShape
        infer_array_cls = np.frombuffer(data_str_cls, dtype=np.float32)
        infer_array_cls.shape = tensor_shape_cls

        tensor_vec_tps = package_vec.tensorVec[2]
        data_str_tps = tensor_vec_tps.dataStr
        tensor_shape_tps = tensor_vec_tps.tensorShape
        infer_array_tps = np.frombuffer(data_str_tps, dtype=np.float32)
        infer_array_tps.shape = tensor_shape_tps

        tensor_vec_fts = package_vec.tensorVec[3]
        data_str_fts = tensor_vec_fts.dataStr
        tensor_shape_fts = tensor_vec_fts.tensorShape
        infer_array_fts = np.frombuffer(data_str_fts, dtype=np.float32)
        infer_array_fts.shape = tensor_shape_fts
        return infer_array_ids, infer_array_cls, infer_array_tps, infer_array_fts


def infer(img_dir, result_path, pipline_path, heatmapaware=True,
          segmentaware=True, FLIP_TEST=True, SHIFT_HEATMAP=True, BatchSize=1):
    ''' start infer '''
    stream = SdkInfer(pipline_path)
    stream.init_stream()
    file_list = os.listdir(img_dir)
    file_list.sort()

    pn_inputs = []
    imgs = []
    centers = []
    scales = []
    batch_cout = 0
    infer_id = 0
    cost_mils = 0.0
    with open(result_path, 'w') as f_write:
        for _, file_name in enumerate(file_list):
            if not file_name.lower().endswith((".jpg", "jpeg")):
                continue
            img_path = os.path.join(img_dir, file_name)
            start_time = datetime.datetime.now()
            img_hwc, pn_input, center, scale = image_proc(img_path)
            imgs.append(img_hwc)
            pn_inputs.append(pn_input)
            batch_cout += 1
            if batch_cout < BatchSize:
                continue
            infer_id = stream.send_package_buf(b'PoseEstNet0', pn_inputs, 0)
            posenet_result = stream.get_result(b'PoseEstNet0', infer_id)
            centers.append(center)
            scales.append(scale)
            if FLIP_TEST:
                input_flipped = np.flip(pn_inputs, 3)
                infer_id = stream.send_package_buf(b'PoseEstNet0', input_flipped, 0)
                outputs_flipped = stream.get_result(b'PoseEstNet0', infer_id)
                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped
                output_flipped = flip_back(
                    np.array(output_flipped), flip_pairs)
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if SHIFT_HEATMAP:  # true
                    output_flipped_copy = output_flipped
                    output_flipped[:, :, :,
                                   1:] = output_flipped_copy[:, :, :, 0:-1]
                posenet_result = (posenet_result + output_flipped) * 0.5

            pn_preds = get_posenet_preds(
                posenet_result, center=centers, scale=scales)
            mt_input, vkpt = gene_mt_input(np.array(
                imgs, np.float32), posenet_result, pn_preds, heatmapaware, segmentaware)

            infer_id = stream.send_package_buf(b'MultiTaskNet0', mt_input, 0)
            infer_id = stream.send_package_buf(b'MultiTaskNet0', vkpt, 1)
            _, cls, tps, _ = stream.get_result(b'MultiTaskNet0', infer_id)
            end_time = datetime.datetime.now()
            cost_mils += (end_time - start_time).microseconds/1000
            pn_inputs = []
            imgs = []
            centers = []
            scales = []
            batch_cout = 0
            for i in range(BatchSize):
                res_list = file_name + ' color:' + '{:d}'.format(np.argmax(cls[i])+1) + \
                    ' type:' + '{:d}'.format(np.argmax(tps[i])+1) + '\n'
                f_write.writelines(res_list)
    print(f'sdk run time: {cost_mils:8.2f} ms; fps: {(1000.0*len(file_list)/cost_mils):8.2f} f/s')
    stream.destroy()


def infer_test(stream, img_path, vkeypt=None, heatmapaware=True, segmentaware=True, FLIP_TEST=True, SHIFT_HEATMAP=True):
    ''' infer single img '''
    img_hwc, pe_input, center, scale = image_proc(img_path)
    pe_input = np.expand_dims(pe_input, axis=0)
    infer_id = stream.send_package_buf(b'PoseEstNet0', pe_input, 0)
    posenet_result = stream.get_result(b'PoseEstNet0', infer_id)

    if FLIP_TEST:
        input_flipped = np.flip(pe_input, 3)
        infer_id = stream.send_package_buf(b'PoseEstNet0', input_flipped, 0)
        outputs_flipped = stream.get_result(b'PoseEstNet0', infer_id)
        if isinstance(outputs_flipped, list):
            output_flipped = outputs_flipped[-1]
        else:
            output_flipped = outputs_flipped
        output_flipped = flip_back(np.array(output_flipped), flip_pairs)
        # output_flipped = np.array(output_flipped)
        # feature is not aligned, shift flipped heatmap for higher accuracy
        if SHIFT_HEATMAP:  # true
            output_flipped_copy = output_flipped
            output_flipped[:, :, :, 1:] = output_flipped_copy[:, :, :, 0:-1]
        posenet_result = (posenet_result + output_flipped) * 0.5

    pn_preds = get_posenet_preds(posenet_result, center=[center], scale=[scale])

    mt_input, vkpts = gene_mt_input(
        np.array([img_hwc]), posenet_result, pn_preds, heatmapaware, segmentaware)

    infer_id = stream.send_package_buf(b'MultiTaskNet0', mt_input, 0)
    if vkeypt is None:
        infer_id = stream.send_package_buf(b'MultiTaskNet0', vkpts, 1)
    else:
        infer_id = stream.send_package_buf(b'MultiTaskNet0', vkeypt, 1)
    return stream.get_result(b'MultiTaskNet0', infer_id)
