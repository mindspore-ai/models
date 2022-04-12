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
""" Model Infer """
import json
import logging
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn, StringVector
from config import config as cfg


class SdkApi:
    """ Class SdkApi """
    INFER_TIMEOUT = cfg.INFER_TIMEOUT
    STREAM_NAME = cfg.STREAM_NAME

    def __init__(self, pipeline_cfg):
        self.pipeline_cfg = pipeline_cfg
        self._stream_api = None
        self._data_input = None
        self._device_id = None

    def init(self):
        """ Initialize Stream """
        with open(self.pipeline_cfg, 'r') as fp:
            self._device_id = int(
                json.loads(fp.read())[self.STREAM_NAME]["stream_config"]
                ["deviceId"])
            print(f"The device id: {self._device_id}.")

        # create api
        self._stream_api = StreamManagerApi()

        # init stream mgr
        ret = self._stream_api.InitManager()
        if ret != 0:
            print(f"Failed to init stream manager, ret={ret}.")
            return False

        # create streams
        with open(self.pipeline_cfg, 'rb') as fp:
            pipe_line = fp.read()

        ret = self._stream_api.CreateMultipleStreams(pipe_line)
        if ret != 0:
            print(f"Failed to create stream, ret={ret}.")
            return False

        self._data_input = MxDataInput()
        return True

    def __del__(self):
        if not self._stream_api:
            return

        self._stream_api.DestroyAllStreams()

    def _send_protobuf(self, stream_name, plugin_id, element_name, buf_type,
                       pkg_list):
        """ Send Stream """
        protobuf = MxProtobufIn()
        protobuf.key = element_name.encode("utf-8")
        protobuf.type = buf_type
        protobuf.protobuf = pkg_list.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)
        err_code = self._stream_api.SendProtobuf(stream_name, plugin_id,
                                                 protobuf_vec)
        if err_code != 0:
            logging.error(
                "Failed to send data to stream, stream_name(%s), plugin_id(%s), element_name(%s), "
                "buf_type(%s), err_code(%s).", stream_name, plugin_id,
                element_name, buf_type, err_code)
            return False
        return True

    def send_tensor_input(self, stream_name, plugin_id, element_name,
                          input_data, input_shape, data_type):
        """ Send Tensor """
        tensor_list = MxpiDataType.MxpiTensorPackageList()
        for i in range(1000):
            data = np.expand_dims(input_data[i, :], 0)
            tensor_pkg = tensor_list.tensorPackageVec.add()
            # init tensor vector
            tensor_vec = tensor_pkg.tensorVec.add()
            tensor_vec.deviceId = self._device_id
            tensor_vec.memType = 0
            tensor_vec.tensorShape.extend(data.shape)
            tensor_vec.tensorDataType = data_type
            tensor_vec.dataStr = data.tobytes()
            tensor_vec.tensorDataSize = data.shape[0]
        print(type(tensor_list))
        buf_type = b"MxTools.MxpiTensorPackageList"
        return self._send_protobuf(stream_name, plugin_id, element_name,
                                   buf_type, tensor_list)

    def get_result(self, stream_name, out_plugin_id=0):
        """ Get Result """
        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer_result = self._stream_api.GetProtobuf(stream_name, 0, keyVec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()
        TensorList = MxpiDataType.MxpiTensorPackageList()
        TensorList.ParseFromString(infer_result[0].messageBuf)
        return TensorList
