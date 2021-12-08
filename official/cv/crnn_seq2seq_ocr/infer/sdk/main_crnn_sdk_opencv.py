# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding:utf-8 -*-
import codecs
import os
import argparse
import numpy as np
import cv2
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn, StringVector
from src.model_utils.config import config
from src.utils import initialize_vocabulary


parser = argparse.ArgumentParser(
    description='tool that takes tfrecord files and extracts all images + labels from it')
parser.add_argument('--imgPath', default='../data/fsns/test', help='path to directory containing tfrecord files')
parser.add_argument('--annoPath', default='../data/fsns/test-anno/image2text.txt',
                    help='path to dir where resulting images shall be saved')

args = parser.parse_args()


def text_standardization(text_in):
    """
    replace some particular characters
    """
    stand_text = text_in.strip()
    stand_text = ' '.join(stand_text.split())
    stand_text = stand_text.replace(u'(', u'（')
    stand_text = stand_text.replace(u')', u'）')
    stand_text = stand_text.replace(u':', u'：')
    return stand_text


def LCS_length(str1, str2):
    """
    calculate longest common sub-sequence between str1 and str2
    """
    if str1 is None or str2 is None:
        return 0

    len1 = len(str1)
    len2 = len(str2)
    if len1 == 0 or len2 == 0:
        return 0

    lcs = [[0 for _ in range(len2 + 1)] for _ in range(2)]
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                lcs[i % 2][j] = lcs[(i - 1) % 2][j - 1] + 1
            else:
                if lcs[i % 2][j - 1] >= lcs[(i - 1) % 2][j]:
                    lcs[i % 2][j] = lcs[i % 2][j - 1]
                else:
                    lcs[i % 2][j] = lcs[(i - 1) % 2][j]

    return lcs[len1 % 2][-1]


def send_source_data(appsrc_id, array_bytes, stream_name, stream_manager, shape):
    """
    Construct the input of the stream,
    send inputs data to a specified stream based on streamName.

    Returns:
        bool: send data success or not
    """

    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    data_input = MxDataInput()
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in shape:
        tensor_vec.tensorShape.append(i)
    data_input.data = array_bytes
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
        print("Failed to send data to stream.")
        return False
    print("Send successfully!")
    return True


def send_image_source_data(appsrc_id, data, stream_name, stream_manager):
    dataInput = MxDataInput()
    dataInput.data = data
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()
    image_width = 512
    image_heigh = 128
    visionVec.visionInfo.format = 1
    visionVec.visionInfo.width = image_width
    visionVec.visionInfo.height = image_heigh
    visionVec.visionInfo.widthAligned = image_width
    visionVec.visionInfo.heightAligned = image_heigh
    visionVec.visionData.deviceId = 0
    visionVec.visionData.memType = 0
    visionVec.visionData.dataStr = dataInput.data
    visionVec.visionData.dataSize = len(dataInput.data)
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = visionList.SerializeToString()
    protobufVec = InProtobufVector()
    protobufVec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobufVec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    print("Send successfully!")
    return True


def crnn_sdk_infer():
    pipeline_path = "../data/config/crnn_seq2seq_ocr.pipeline"
    stream_name = b'detection'
    dir_name = args.imgPath
    text_path = args.annoPath
    _, rev_vocab = initialize_vocabulary("../../general_chars.txt")
    num_total_word = 0
    num_correct_word = 0
    num_correct_char = 0
    num_total_char = 0

    res_dir_name = 'results'
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    anno_text = {}
    anno_file = open(text_path, 'r').readlines()
    for line in anno_file:
        file_name = line.split('\t')[0]
        labels = line.split('\t')[1].split('\n')[0]
        anno_text[file_name] = labels

    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    sos_id = config.characters_dictionary.go_id
    eos_id = config.characters_dictionary.eos_id
    decoder_input = (np.ones((config.eval_batch_size, 1)) * sos_id).astype(np.int32)
    decoder_hidden = np.zeros((1, config.eval_batch_size, config.decoder_hidden_size),
                              dtype=np.float16)

    file_list = os.listdir(dir_name)
    correct_file = './results/result_correct.txt'
    incorrect_file = './results/result_incorrect.txt'
    with codecs.open(correct_file, 'w', encoding='utf-8') as fp_output_correct, \
            codecs.open(incorrect_file, 'w', encoding='utf-8') as fp_output_incorrect:

        for file_name in file_list:
            print(file_name)
            if file_name.endswith(".JPG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                file_path = os.path.join(dir_name, file_name)

                img = cv2.imread(file_path)
                img = cv2.resize(img, (512, 128)).astype(np.float32)
                cv2.normalize(img, img, -1, 1, cv2.NORM_MINMAX)
                img = img.transpose((2, 0, 1))
                img = img.reshape(config.eval_batch_size, img.shape[0], img.shape[1], img.shape[2])

                send_image_source_data(0, img.tobytes(), stream_name, stream_manager_api)
                send_source_data(1, decoder_input.tobytes(), stream_name, stream_manager_api, decoder_input.shape)
                send_source_data(2, decoder_hidden.tobytes(), stream_name, stream_manager_api, decoder_hidden.shape)

                key_vec = StringVector()
                key_vec.push_back(b'mxpi_tensorinfer0')
                infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
                if infer_result.size() == 0:
                    print("inferResult is null")
                    break
                if infer_result[0].errorCode != 0:
                    print("GetProtobuf error. errorCode=%d" % (infer_result[0].errorCode))
                    break
                result = MxpiDataType.MxpiTensorPackageList()
                result.ParseFromString(infer_result[0].messageBuf)

                decoded_words = []
                for tensor in result.tensorPackageVec[0].tensorVec:
                    idx = np.frombuffer(tensor.dataStr, dtype=np.int32)[0]
                    if idx == eos_id:
                        break
                    decoded_words.append(rev_vocab[idx])
                text = anno_text[file_name]
                text = text_standardization(text)
                predict = text_standardization("".join(decoded_words))

                if predict == text:
                    num_correct_word += 1
                    fp_output_correct.write('\t\t' + text + '\n')
                    fp_output_correct.write('\t\t' + predict + '\n\n')

                else:
                    fp_output_incorrect.write('\t\t' + text + '\n')
                    fp_output_incorrect.write('\t\t' + predict + '\n\n')

                num_total_word += 1
                num_correct_char += 2 * LCS_length(text, predict)
                num_total_char += len(text) + len(predict)

    print('\nnum of correct characters = %d' % (num_correct_char))
    print('\nnum of total characters = %d' % (num_total_char))
    print('\nnum of correct words = %d' % (num_correct_word))
    print('\nnum of total words = %d' % (num_total_word))
    print('\ncharacter precision = %f' % (float(num_correct_char) / num_total_char))
    print('\nAnnotation precision precision = %f' % (float(num_correct_word) / num_total_word))

    with open("./results/eval_sdk.log", 'w') as f:
        f.write('num of correct characters = {}\n'.format(num_correct_char))
        f.write('num of total characters = {}\n'.format(num_total_char))
        f.write('num of correct words = {}\n'.format(num_correct_word))
        f.write('num of total words = {}\n'.format(num_total_word))
        f.write('character precision = {}\n'.format(float(num_correct_char) / num_total_char))
        f.write('Annotation precision precision = {}\n'.format(float(num_correct_word) / num_total_word))

    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    crnn_sdk_infer()
