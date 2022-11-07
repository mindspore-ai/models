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
""" main.py """
import os
import argparse
import math
import copy
from pathlib import Path
from StreamManagerApi import MxDataInput, InProtobufVector, MxProtobufIn, StringVector, StreamManagerApi
import MxpiDataType_pb2 as MxpiDataType
import numpy as np

FRAME_SIZE = 160
NB_FEATURES = 36
NB_USED_FEATURES = 20
ORDER = 16
RNN_UNITS1 = 384
RNN_UNITS2 = 16
scale = 255.0/32768.0 # 8-bit to 16-bit range ratio
scale_1 = 32768.0/255.0 # 16-bit to 8-bit range ratio

def parse_args(parsers):
    """
    Parse commandline arguments.
    """
    parsers.add_argument('--feature_path', type=Path, default="../data/eval-data", help='input data path')
    parser.add_argument("--output_path", type=Path, default="../result/sdk", help="output path")
    parser.add_argument("--max_len", type=int, default=500)
    return parsers


def ulaw2lin(u):
    """ Transform signal from 8-bit mu-law quantized to linear """
    u = u - 128
    s = np.sign(u)
    u = np.abs(u)
    return s*scale_1*(np.exp(u/128.*math.log(256))-1)


def lin2ulaw(x):
    """ Transform signal from linear to 8-bit mu-law quantized """
    s = np.sign(x)
    x = np.abs(x)
    u = (s*(128*np.log(1+scale*x)/math.log(256)))
    u = np.clip(128 + np.round(u), 0, 255)
    return u.astype('int16')

def send_data(inputs, name, stream_manager2):
    inPluginId = 0
    for tensor in inputs:
        tensorPackageList = MxpiDataType.MxpiTensorPackageList()
        tensorPackage = tensorPackageList.tensorPackageVec.add()
        array_bytes = tensor.tobytes()
        dataInput = MxDataInput()
        dataInput.data = array_bytes
        tensorVec = tensorPackage.tensorVec.add()
        tensorVec.deviceId = 0
        tensorVec.memType = 0
        for k in tensor.shape:
            tensorVec.tensorShape.append(k)
        tensorVec.dataStr = dataInput.data
        tensorVec.tensorDataSize = len(array_bytes)

        key = "appsrc{}".format(inPluginId).encode('utf-8')
        protobufVec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensorPackageList.SerializeToString()
        protobufVec.push_back(protobuf)

        ret_ = stream_manager2.SendProtobuf(name, key, protobufVec)
        inPluginId += 1
        if ret_ < 0:
            print("Failed to send data to stream.")
            exit()

def infer(features, out_file, stream_manager_):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
    bool: send data success or not
    """
    c = 0
    nb_frames = 1
    feature_chunk_size = 500
    pcm_chunk_size = FRAME_SIZE * feature_chunk_size

    features = np.reshape(features, (nb_frames, feature_chunk_size, NB_FEATURES))
    periods = (.1 + 50 * features[:, :, 18:19] + 100).astype('int32')

    pcm = np.zeros((nb_frames * pcm_chunk_size,))
    fexc = np.zeros((1, 1, 3), dtype='int32') + 128
    state1 = np.zeros((1, 1, RNN_UNITS1)).astype(np.float32)
    state2 = np.zeros((1, 1, RNN_UNITS2)).astype(np.float32)
    enc_input1 = features[c : c +1, :, :NB_USED_FEATURES].astype(np.float32)
    enc_input2 = periods[c : c + 1, :, :]
    mem = 0.
    coef = 0.85

    with open(out_file, 'wb') as fout:
        enc_inputs = [enc_input1, enc_input2]
        send_data(enc_inputs, b'lpcnet_encoder', stream_manager_)
        # Obtain the inference result by specifying streamName and uniqueId.
        keyVec = StringVector()
        keyVec.push_back(b'mxpi_encoder0')
        infer_result = stream_manager_.GetProtobuf(b'lpcnet_encoder', 0, keyVec)
        if infer_result.size() == 0:
            print("inferResult is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()
        # get infer result
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        cfeat = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32).reshape(1, 500, 128) #fp32
        # start decoder
        skip = ORDER + 1
        for fr in range(0, feature_chunk_size):
            f = c * feature_chunk_size + fr
            a = features[c, fr, NB_FEATURES - ORDER:]
            for i in range(skip, FRAME_SIZE):
                pred = -sum(a * pcm[f * FRAME_SIZE + i - 1:f * FRAME_SIZE + i - ORDER - 1:-1])
                fexc[0, 0, 1] = lin2ulaw(pred)
                dec_inputs = [fexc, cfeat[:, fr:fr + 1, :], state1, state2]
                send_data(dec_inputs, b'lpcnet_decoder', stream_manager_)
                # Obtain the inference result by specifying streamName and uniqueId.
                keyVec = StringVector()
                keyVec.push_back(b'mxpi_decoder0')
                infer_result = stream_manager_.GetProtobuf(b'lpcnet_decoder', 0, keyVec)
                if infer_result.size() == 0:
                    print("inferResult is null")
                    exit()
                if infer_result[0].errorCode != 0:
                    print("GetProtobuf error. errorCode=%d" % (
                        infer_result[0].errorCode))
                    exit()
                # get infer result
                result = MxpiDataType.MxpiTensorPackageList()
                result.ParseFromString(infer_result[0].messageBuf)
                # convert the inference result to Numpy array
                p = copy.copy(np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32))
                p = p.reshape(1, 1, 256)
                p = p.astype(np.float64)
                state1 = copy.copy(np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32))
                state1 = state1.reshape(1, 1, 384)
                state2 = copy.copy(np.frombuffer(result.tensorPackageVec[0].tensorVec[2].dataStr, dtype=np.float32))
                state2 = state2.reshape(1, 1, 16)
                # lower the temperature for voiced frames to reduce noisiness
                p *= np.power(p, np.maximum(0, 1.5 * features[c, fr, 19] - .5))
                p = p / (1e-18 + np.sum(p))
                p = np.maximum(p - 0.002, 0).astype('float64')
                p = p / (1e-8 + np.sum(p))
                fexc[0, 0, 2] = np.argmax(np.random.multinomial(1, p[0, 0, :], 1))
                pcm[f * FRAME_SIZE + i] = pred + ulaw2lin(fexc[0, 0, 2])
                fexc[0, 0, 0] = lin2ulaw(pcm[f * FRAME_SIZE + i])
                mem = coef * mem + pcm[f * FRAME_SIZE + i]
                np.array([np.round(mem)], dtype='int16').tofile(fout)
            skip = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Om lpcnet Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../data/config/lpcnet.pipeline", 'rb') as f1:
        pipeline = f1.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    tst_dir = args.feature_path
    out_dir = args.output_path
    max_len = args.max_len
    loop = 0
    data = []
    for _f in out_dir.glob("*.pcm"):
        data.append(_f.stem)
    for _f in tst_dir.glob('*.f32'):
        if _f.stem in data:
            continue
        loop += 1
        _feature_file = tst_dir / (_f.stem + '.f32')
        _out_file = out_dir / (_f.stem + '.pcm')
        features_ = np.fromfile(_feature_file, dtype='float32')
        features_ = np.reshape(features_, (-1, NB_FEATURES))
        feature_chunk_size_ = features_.shape[0]
        if feature_chunk_size_ < 500:
            zeros = np.zeros((500 - feature_chunk_size_, 36))
            features_ = np.concatenate((features_, zeros), 0)
        else:
            features_ = features_[:500, :]
        infer(features_, _out_file, stream_manager)
    print("loop is", loop)

    # destroy streams
    stream_manager.DestroyAllStreams()
