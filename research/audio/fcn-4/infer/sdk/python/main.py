# coding=utf-8

"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import os
import sys
import librosa
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, InProtobufVector, MxProtobufIn, StringVector, MxDataInput

def compute_melgram(audio_path):
    """
    extract melgram feature from the audio and save as numpy array

    Args:
        audio_path (str): root path to the audio clip.
    Returns:
        numpy array.

    """
    SR = 12000 # sample rate
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..
    try:
        src, _ = librosa.load(audio_path, sr=SR)  # whole signal
    except EOFError:
        print("file was damaged:", audio_path)
        print("now skip it!")
        return np.array([0])
    except FileNotFoundError:
        print("cant load file:", audio_path)
        print("now skip it!")
        return np.array([0])
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)
    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.core.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    mel_feature = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS))
    mel_feature = mel_feature[np.newaxis, np.newaxis, :]
    mel_feature = np.array(mel_feature, dtype=np.float32)
    return mel_feature

if __name__ == '__main__':
    streamManagerApi = StreamManagerApi() # init stream manager
    ret = streamManagerApi.InitManager()
    streamName = b'im_fcn4'
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # create streams by pipeline config file
    pipeline_path = "../pipeline/fcn-4.pipeline"
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # Construct the input of the stream
    data_input = MxDataInput()

    root_audio_path = sys.argv[1]
    result_path = sys.argv[2]

    dirname = os.listdir(root_audio_path)
    for d in dirname:
        base_audio_path = os.listdir(os.path.join(root_audio_path, d))
        for f in base_audio_path:
            abs_audio_path = os.path.join(root_audio_path, d, f)
            tensor = compute_melgram(abs_audio_path)
            if len(tensor.shape) < 4:
                continue
            print("-"*40)
            print("Absolute audio path: ", abs_audio_path)
            print("tensor shape: ", tensor.shape)
            inPluginId = 0
            tensorPackageList = MxpiDataType.MxpiTensorPackageList()
            tensorPackage = tensorPackageList.tensorPackageVec.add()
            # add feature data begin
            array_bytes = tensor.tobytes()
            dataInput = MxDataInput()
            dataInput.data = array_bytes
            tensorVec = tensorPackage.tensorVec.add()
            tensorVec.deviceId = 0
            tensorVec.memType = 0
            for i in tensor.shape:
                tensorVec.tensorShape.append(i)
            tensorVec.dataStr = dataInput.data
            # compute the number of bytes of feature data
            tensorVec.tensorDataSize = len(array_bytes)
            # add feature data end
            key = "appsrc{}".format(inPluginId).encode('utf-8')
            protobufVec = InProtobufVector()
            protobuf = MxProtobufIn()
            protobuf.key = key
            protobuf.type = b'MxTools.MxpiTensorPackageList'
            protobuf.protobuf = tensorPackageList.SerializeToString()
            protobufVec.push_back(protobuf)

            uniqueId = streamManagerApi.SendProtobuf(streamName, inPluginId, protobufVec)
            if uniqueId < 0:
                print("Failed to send data to stream.")
                exit()

            keyVec = StringVector()
            keyVec.push_back(b'mxpi_tensorinfer0')
            # get inference result
            inferResult = streamManagerApi.GetProtobuf(streamName, 0, keyVec)

            if inferResult.size() == 0:
                print("inferResult is null")
                exit()
            if inferResult[0].errorCode != 0:
                print("GetProtobuf error. errorCode=%d" % (
                    inferResult[0].errorCode))
                exit()
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(inferResult[0].messageBuf)
            # convert the inference result to Numpy array
            # speaker embedding
            res = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='<f4')
            # post_processing
            print("="*20, " result ", "="*20)
            print(res)
            if not os.path.exists(os.path.join(result_path, d)):
                os.makedirs(os.path.join(result_path, d))
            save_name = os.path.join(result_path, d, f[:-4]+'.txt')
            np.savetxt(save_name, np.array([res]), fmt='%s', delimiter=',')
    # destroy streams
    streamManagerApi.DestroyAllStreams()
