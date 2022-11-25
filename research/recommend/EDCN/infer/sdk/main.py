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


import time
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn


def send_data(input_stream_name, input_plugin_id, data_np):
    vision_list = MxpiDataType.MxpiVisionList()
    vision_vec = vision_list.visionVec.add()
    vision_vec.visionInfo.format = 0
    vision_vec.visionData.memType = 0
    vision_vec.visionData.dataStr = data_np.tobytes()
    vision_vec.visionData.dataSize = len(data_np)

    in_plugin_id = input_plugin_id
    protobuf = MxProtobufIn()
    key_str = "appsrc{}".format(input_plugin_id).encode('utf-8')
    protobuf.key = key_str
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = vision_list.SerializeToString()
    protobuf_vec = InProtobufVector()
    protobuf_vec.push_back(protobuf)

    unique_id = stream_manager_api.SendProtobuf(input_stream_name, in_plugin_id, protobuf_vec)

    return unique_id


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("[INFO] Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    print("[INFO] Init Stream manager successfully!")

    # create streams by pipeline config file
    with open("../data/config/edcn.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("[INFO] Failed to create Stream, ret=%s" % str(ret))
        exit()
    print("[INFO] Create Stream successfully!")

    label = np.loadtxt('../data/input/label.txt', dtype=np.float32, delimiter="\t")
    ids = np.loadtxt('../data/input/feat_ids.txt', dtype=np.int32, delimiter="\t")
    wts = np.loadtxt('../data/input/feat_vals.txt', dtype=np.float32, delimiter="\t")
    rows = label.shape[0]
    p_label = []
    t_label = []
    prob = []
    label_list = []
    stream_name = b'im_edcn'
    infer_total_time = 0

    for i in range(rows):
        print(i)
        feat_ids = ids[i]
        feat_vals = wts[i]
        labels = label[i]
        label_list.append(labels)
        tlabel = np.array(label_list, dtype=np.float32)
        inplugin_id = 0
        unique_id1 = send_data(stream_name, inplugin_id, feat_ids)
        if unique_id1 < 0:
            print("[INFO] Failed to send data to stream.")
            exit()
        print("[INFO] Send data to stream successfully!")

        inplugin_id = 1
        unique_id2 = send_data(stream_name, inplugin_id, feat_vals)
        if unique_id2 < 0:
            print("[INFO] Failed to send data to stream.")
            exit()
        print("[INFO] Send data to stream successfully!")

        inplugin_id = 2
        unique_id3 = send_data(stream_name, inplugin_id, tlabel)
        if unique_id3 < 0:
            print("[INFO] Failed to send data to stream.")
            exit()
        print("[INFO] Send data to stream successfully!")
        # Obtain the inference result by specifying streamName and uniqueId.

        keyVec = StringVector()
        keyVec.push_back(b'mxpi_tensorinfer0')
        start_time = time.time()

        # infer_result = stream_manager_api.GetProtobuf(stream_name, 0,keyVec)
        data_source_vector = StringVector()
        data_source_vector.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, keyVec)
        infer_total_time += time.time() - start_time

        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        result_np = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        result_np1 = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)
        result_np2 = np.frombuffer(result.tensorPackageVec[0].tensorVec[2].dataStr, dtype=np.float32)
        print(result_np, result_np1, result_np2)

        if infer_result.size() == 0:
            print("[INFO] infer result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("[INFO] GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" %
                  (infer_result[0].errorCode, infer_result[0].data.decode()))
            exit()
        print("[INFO] Get result successfully!")

        # Transform result from buffer to numpy
        label_list.clear()
        prob.append(result_np1)
        if result_np1 < 0.5:
            p_label.append(0)
        else:
            p_label.append(1)
        t_label.append(result_np2)

    np.savetxt('t_label.txt', t_label)
    np.savetxt('p_label.txt', p_label)
    np.savetxt('prob.txt', prob)
    fo1 = open("metric.txt", "w")
    fo1.write("Number of samples:%d\n"%(rows))
    fo1.write("Infer total time:%f\n"%(infer_total_time))
    fo1.write("Average infer time:%f\n"%(infer_total_time/rows))
    fo1.close()
    print('<<========  Infer Metric ========>>')
    print("Number of samples:%d"%(rows))
    print("Infer total time:%f"%(infer_total_time))
    print("Average infer time:%f\n"%(infer_total_time/rows))
    print('<<===============================>>')



    # destroy streams
    stream_manager_api.DestroyAllStreams()
