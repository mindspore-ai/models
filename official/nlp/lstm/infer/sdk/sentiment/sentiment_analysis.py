# coding = utf-8
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License  (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import MxpiDataType_pb2 as MxpiDataType
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, \
    MxProtobufIn, StringVector


def send_source_data(appsrc_id, raw_data, stream_name, stream_manager):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on stream name
    :param appsrc_id: corresponding appsrc id
    :param raw_data: data which need to send into stream
    :param stream_name: the name of infer stream that needs to operate
    :param stream_manager: the manager of infer streams
    :return bool: send data success or not
    """
    tensor = np.array(raw_data).astype(np.int32)
    # expand dim when raw data only one-dimension
    word_vector_length = 500
    if len(raw_data.shape) == 1 and raw_data.shape[0] == word_vector_length:
        tensor = np.expand_dims(tensor, 0)

    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    array_bytes = tensor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tensor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = data_input.data
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
        print("Failed to send data to stream, ret = {}.".format(ret))
        return False
    return True


def infer(stream_manager, stream_name, in_plugin_id, data_input):
    """
    send data into infer stream and get infer result
    :param stream_manager: the manager of infer streams
    :param stream_name: the name of infer stream that needs to operate
    :param in_plugin_id: ID of the plug-in that needs to send data
    :param data_input: data that needs to send into infer stream
    :return: infer results
    """
    # Inputs data to a specified stream based on stream name
    send_success = send_source_data(in_plugin_id, data_input, stream_name, stream_manager)
    if not send_success:
        print('Failed to send data to stream')
        return None

    # construct output plugin vector
    plugin_names = [b"mxpi_tensorinfer0"]
    plugin_vec = StringVector()
    for key in plugin_names:
        plugin_vec.push_back(key)

    # get plugin output data
    infer_result = stream_manager.GetProtobuf(stream_name, in_plugin_id, plugin_vec)

    # check whether the inferred results is valid
    infer_result_valid = True
    if infer_result.size() == 0:
        infer_result_valid = False
        print('unable to get effective infer results, please check the stream log for details.')
    elif infer_result[0].errorCode != 0:
        infer_result_valid = False
        print('GetProtobuf error. errorCode = {}, errorMsg= {}.'.format(
            infer_result[0].errorCode, infer_result[0].data.decode()))

    if not infer_result_valid:
        return None

    # get mxpi_tensorinfer0 output data
    infer_result_list = MxpiDataType.MxpiTensorPackageList()
    infer_result_list.ParseFromString(infer_result[0].messageBuf)
    # get infer result data str
    output_data_str = infer_result_list.tensorPackageVec[0].tensorVec[0].dataStr
    output_data = np.frombuffer(output_data_str, dtype=np.float32)
    # get predict probability with softmax function
    pred_prop = np.exp(output_data) / sum(np.exp(output_data))
    # get predict result
    pred_result = np.argmax(pred_prop, axis=0)

    # print result
    print('origin output: ', output_data)
    print('pred prob: ', pred_prop)
    print('pred result: ', pred_result)

    return pred_result


def senti_analysis(sentence_word_vector, sentiment_pipeline_path, is_batch=False, batch_size=64):
    """
    sentiment analysis of review
    :param sentence_word_vector: word vectors of reviews
    :param sentiment_pipeline_path: pipeline path
    :param is_batch: whether batch data
    :param batch_size: batch size
    :return: model inference result
    """
    stream_manager = StreamManagerApi()

    # init stream manager
    ret = stream_manager.InitManager()
    if ret != 0:
        print('Failed to init Stream manager, ret = {}.'.format(ret))
        return None

    if os.path.exists(sentiment_pipeline_path) != 1:
        print('pipeline {} not exist.'.format(sentiment_pipeline_path))
        return None

    # create streams by pipeline config file
    with open(sentiment_pipeline_path, 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pipeline_str = pipeline
    ret = stream_manager.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print('Failed to create Stream, ret = {}.'.format(ret))
        return None

    # config
    stream_name = b'sentiment'
    in_plugin_id = 0

    # infer results
    sentiment_result = np.empty(shape=(0, 1))

    # Construct the input of the stream
    if not is_batch:
        # model infer
        result = infer(stream_manager, stream_name, in_plugin_id, sentence_word_vector)

        # check infer result
        if result is None:
            print('sentiment model infer error.')
            # destroy streams
            stream_manager.DestroyAllStreams()
            return None
        sentiment_result = result
    else:
        batch_idx = 0
        processed = 0
        total = sentence_word_vector.shape[0]
        # batch infer not support, force assign batch size to 1
        batch_size = 1

        percent_mask = 100.
        for i in range(0, total, batch_size):
            batch_idx += 1
            if i + batch_size > total:
                word_vector = sentence_word_vector[i:]
            else:
                word_vector = sentence_word_vector[i:(i + batch_size)]

            # model infer
            result = infer(stream_manager, stream_name, in_plugin_id, word_vector)

            # check infer result
            if result is None:
                print('sentiment model infer error on {}-th batch sentences.'.format(batch_idx))
                break

            # save infer result
            processed += word_vector.shape[0]
            sentiment_result = np.vstack([sentiment_result, [result]])
            print('batch size: {}, processed {}-th batch sentences, '
                  '[{}/{} ({:.0f}%)].'.format(batch_size, batch_idx,
                                              processed, total,
                                              percent_mask * processed / total))

    # destroy streams
    stream_manager.DestroyAllStreams()

    return sentiment_result
