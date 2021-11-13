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
import io
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
from StreamManagerApi import MxDataInput, StringVector, StreamManagerApi, InProtobufVector, MxProtobufIn

if __name__ == '__main__':
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/faceattribute.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    # label_txt index
    label_txt = sys.argv[1]

    # Define the required later
    total_data_num_age, total_data_num_gen, total_data_num_mask = 0, 0, 0
    age_num, gen_num, mask_num = 0, 0, 0
    gen_tp_num, mask_tp_num, gen_fp_num = 0, 0, 0
    mask_fp_num, gen_fn_num, mask_fn_num = 0, 0, 0
    with open(label_txt, 'r') as ft:
        lines = ft.readlines()
        for line in lines:
            sline = line.strip().split(" ")
            image_file = sline[0]
            imgName = image_file.split('/')
            # Get the name of the image
            file_name = imgName[-1]
            gt_age, gt_gen, gt_mask = int(sline[1]), int(sline[2]), int(sline[3])
            # Get the total path to the image
            file_path = sline[0]
            if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
                continue
            print("processing img ", file_path)
            with open(file_path, 'rb') as f:
                img = f.read()
            # Reproduce the preprocessing operations in the eval.py file
            data = io.BytesIO(img)
            img = Image.open(data)
            img = img.convert('RGB')
            img = np.asarray(img)
            img = cv2.resize(img, (112, 112))
            img = (img - 127.5) / 127.5
            img = img.astype(np.float32)
            img = img.transpose(2, 0, 1)
            tensor = img[None]

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

            empty_data = []
            stream_name = b'im_resnet18'

            in_plugin_id = 0
            uniqueId = stream_manager_api.SendProtobuf(stream_name, inPluginId, protobufVec)
            if uniqueId < 0:
                print("Failed to send data to stream.")
                exit()
            keyVec = StringVector()
            keyVec.push_back(b'mxpi_tensorinfer0')
            # get inference result
            inferResult = stream_manager_api.GetProtobuf(stream_name, 0, keyVec)

            if inferResult.size() == 0:
                print("inferResult is null")
                exit()
            if inferResult[0].errorCode != 0:
                print("GetProtobuf error. errorCode=%d" % (
                    inferResult[0].errorCode))
                exit()
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(inferResult[0].messageBuf)
            resAge = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype='float32')
            resGender = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype='float32')
            resMask = np.frombuffer(result.tensorPackageVec[0].tensorVec[2].dataStr, dtype='float32')

            age_result_np = np.empty(shape=(1, 9))
            flag = 0
            for item in resAge:
                age_result_np[0][flag] = item
                flag += 1

            gen_result_np = np.empty(shape=(1, 2))

            flag = 0
            for item in resGender:
                gen_result_np[0][flag] = item
                flag += 1
            mask_result_np = np.empty(shape=(1, 2))

            flag = 0
            for item in resMask:
                mask_result_np[0][flag] = item
                flag += 1
            age_prob = age_result_np[0].tolist()
            gen_prob = gen_result_np[0].tolist()
            mask_prob = mask_result_np[0].tolist()
            age = age_prob.index(max(age_prob))
            gen = gen_prob.index(max(gen_prob))
            mask = mask_prob.index(max(mask_prob))
            if gt_age == age:
                age_num += 1
            if gt_gen == gen:
                gen_num += 1
            if gt_mask == mask:
                mask_num += 1

            if gen == 1:
                if gt_gen == 1:
                    gen_tp_num += 1
                elif gt_gen == 0:
                    gen_fp_num += 1
            elif gen == 0 and gt_gen == 1:
                gen_fn_num += 1

            if gt_mask == 1 and mask == 1:
                mask_tp_num += 1
            if gt_mask == 0 and mask == 1:
                mask_fp_num += 1
            if gt_mask == 1 and mask == 0:
                mask_fn_num += 1

            if gt_age != -1:
                total_data_num_age += 1
            if gt_gen != -1:
                total_data_num_gen += 1
            if gt_mask != -1:
                total_data_num_mask += 1
    # The following package is not recommended if it has too many parameters
    print("age_num is ", age_num)
    age_accuracy = float(age_num) / float(total_data_num_age)

    gen_precision = float(gen_tp_num) / (float(gen_tp_num) + float(gen_fp_num))
    gen_recall = float(gen_tp_num) / (float(gen_tp_num) + float(gen_fn_num))
    gen_accuracy = float(gen_num) / float(total_data_num_gen)
    gen_f1 = 2. * gen_precision * gen_recall / (gen_precision + gen_recall)

    print("mask_tp_num is " + str(mask_tp_num))
    print("mask_fn_num is " + str(mask_fn_num))
    print("mask_fp_num is " + str(mask_fp_num))
    mask_precision = float(mask_tp_num) / (float(mask_tp_num) + float(mask_fp_num))
    mask_recall = float(mask_tp_num) / (float(mask_tp_num) + float(mask_fn_num))
    mask_accuracy = float(mask_num) / float(total_data_num_mask)
    mask_f1 = 2. * mask_precision * mask_recall / (mask_precision + mask_recall)

    print('total age num: ', total_data_num_age)
    print('total gen num: ', total_data_num_gen)
    print('total mask num: ', total_data_num_mask)
    print('age accuracy: ', age_accuracy)
    print('gen accuracy: ', gen_accuracy)
    print('mask accuracy: ', mask_accuracy)
    print('gen precision: ', gen_precision)
    print('gen recall: ', gen_recall)
    print('gen f1: ', gen_f1)
    print('mask precision: ', mask_precision)
    print('mask recall: ', mask_recall)
    print('mask f1: ', mask_f1)

    result_txt = os.path.join('./result.txt')
    if os.path.exists(result_txt):
        os.remove(result_txt)
    with open(result_txt, 'a') as ft:
        ft.write('total age num: {}\n'.format(total_data_num_age))
        ft.write('total gen num: {}\n'.format(total_data_num_gen))
        ft.write('total mask num: {}\n'.format(total_data_num_mask))
        ft.write('age accuracy: {}\n'.format(age_accuracy))
        ft.write('gen accuracy: {}\n'.format(gen_accuracy))
        ft.write('mask accuracy: {}\n'.format(mask_accuracy))
        ft.write('gen precision: {}\n'.format(gen_precision))
        ft.write('gen recall: {}\n'.format(gen_recall))
        ft.write('gen f1: {}\n'.format(gen_f1))
        ft.write('mask precision: {}\n'.format(mask_precision))
        ft.write('mask recall: {}\n'.format(mask_recall))
        ft.write('mask f1: {}\n'.format(mask_f1))
    # destroy streams
    stream_manager_api.DestroyAllStreams()
