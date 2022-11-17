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
""" inference for wgan."""
import json
import os
import datetime
import argparse
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
from StreamManagerApi import StreamManagerApi, InProtobufVector, \
    MxProtobufIn, StringVector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='path to generator config .json file')
    parser.add_argument('--nimages', required=True, type=int,
                        help="number of images to generate. (only support nimages=1)", default=1)
    parser.add_argument('--save_path', required=True, type=str, help="save result path", default='../data/result/')
    args_opt = parser.parse_args()
    return args_opt


def generate_data_to_stream(Args, streamName, in_id, stream_m):
    """
    generate data.
    """
    with open(Args.config, 'r') as gencfg:
        generator_c = json.loads(gencfg.read())
    nz = generator_c["nz"]
    # initialize noise
    fixed_noise = np.random.normal(size=[Args.nimages, nz, 1, 1]).astype(np.float32)
    print("*" * 20, "generate input noise finished", "*" * 20)

    input_noise_bytes = fixed_noise.tobytes()
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    # init tensor vec
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = in_id
    tensor_vec.memType = 0
    input_shape = [Args.nimages, nz, 1, 1]
    tensor_vec.tensorShape.extend(input_shape)
    tensor_vec.tensorDataType = 0
    tensor_vec.dataStr = input_noise_bytes
    tensor_vec.tensorDataSize = len(input_noise_bytes)

    protobuf = MxProtobufIn()
    protobuf.key = "appsrc0".encode('utf-8')
    protobuf.type = b"MxTools.MxpiTensorPackageList"
    protobuf.protobuf = tensor_package_list.SerializeToString()

    protobuf_vec = InProtobufVector()
    protobuf_vec.push_back(protobuf)

    rett = stream_m.SendProtobuf(streamName, 0, protobuf_vec)
    if rett < 0:
        print("Failed to send data to stream.")
        return False
    return True


if __name__ == '__main__':
    # init stream manager
    args = parse_args()
    with open(args.config, 'r') as generatecfg:
        generator_config = json.loads(generatecfg.read())

    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(os.path.realpath("./config/WGAN.pipeline"), 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    stream_name = b'im_wgan'
    in_plugin_id = 0
    generate_data_to_stream(args, stream_name, in_plugin_id, stream_manager_api)
    start_time = datetime.datetime.now()

    # begin to infer and get infer result
    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    infer_result = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, key_vec)
    end_time = datetime.datetime.now()

    if infer_result.size() == 0:
        print("inferResult is null")
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))

    # postprocess
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
    res = res.reshape(args.nimages, generator_config["nc"], generator_config["imageSize"],
                      generator_config["imageSize"])
    res = np.multiply(res, 0.5 * 255)
    res = np.add(res, 0.5 * 255)
    res = res[0].astype(np.uint8).transpose((1, 2, 0))
    img = Image.fromarray(res)
    now_time = datetime.datetime.now()

    img_name = str(now_time.hour) + str(now_time.minute) + str(now_time.second)

    img.save(os.path.join(args.save_path, "generated_%s.png" % img_name))

    # destroy streams
    stream_manager_api.DestroyAllStreams()
