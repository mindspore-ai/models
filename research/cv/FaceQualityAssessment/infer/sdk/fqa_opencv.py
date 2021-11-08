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
============================================================================
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np
from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi


def parse_arg():
    """Return the cmd input args"""
    parser = argparse.ArgumentParser(description="FQA infer")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="the directory of dataset")
    parser.add_argument("-p", "--pipeline", type=str, required=True, help="the path of .pipeline file")
    parser.add_argument("-o", "--output", type=str, default="", help="the path of pipeline file")
    return parser.parse_args()


def get_dataset(path):
    """
    Summary.

    yield a image absolutely path when called

    Args:
        path(string): dataset path.
    """
    for _, _, files in os.walk(path):
        for file_name in files:
            if file_name.endswith('jpg'):
                yield os.path.join(path, file_name)
        break


def get_stream_manager(pipeline_path):
    """
    Summary.

    get stream manager, initialize the stream manager api depend on pipeline

    Args:
        pipeline_path(string): mindx-sdk inference pipeline path.

    Returns:
        ?, stream manager api.
    """
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        return -1

    with open(pipeline_path, 'rb') as f:
        pipeline_content = f.read()

    ret = stream_manager_api.CreateMultipleStreams(pipeline_content)
    if ret != 0:
        print("Failed to create stream, ret=%s" % str(ret))
        return -1
    return stream_manager_api


def do_infer_image(stream_manager_api, image_path):
    """
    Summary.

    do inference

    Args:
        stream_manager_api(?): stream manager.
        image_path(string): inferred image path.

    Returns:
        list, reshaped inference result.
    """
    stream_name = b'face_quality_assessment'
    data_input = MxDataInput()
    with open(image_path, 'rb') as f:
        data_input.data = f.read()

    unique_id = stream_manager_api.SendData(stream_name, 0, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        return -1

    infer_result = stream_manager_api.GetResult(stream_name, unique_id)
    if infer_result.errorCode != 0:
        print(f"GetResult error. errorCode={infer_result.errorCode},"
              f"errorMsg={infer_result.data.decode()}")
        return -1

    infer_result_json = json.loads(infer_result.data.decode())
    content = json.loads(infer_result_json['metaData'][0]['content'])
    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][0]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    out_eul = np.frombuffer(base64.b64decode(data_str), dtype=np.float32) * 90
    out_eul = np.reshape(out_eul, tensor_shape[1:])

    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][1]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    heatmap = np.frombuffer(base64.b64decode(data_str), dtype=np.float32)
    heatmap = np.reshape(heatmap, tensor_shape[1:])
    return out_eul, heatmap


def read_ground_truth(img_path):
    """
    Summary.

    Read ground truth from txt path corresponding to image path

    Args:
        img_path(string): image path.

    Returns:
        list, [YAW,PITCH,ROLL], key point list, bool value(is ground truth valid).
    """
    txt_path = ""
    if img_path.endswith('jpg'):
        txt_path = img_path.replace('jpg', 'txt')
    else:
        print("[ERROR], image format is invalid, REQUIRED .jpg")
        return -1
    if os.path.exists(txt_path):
        img_ori = cv2.imread(img_path)
        x_length = img_ori.shape[1]
        y_length = img_ori.shape[0]
        txt_line = open(txt_path).readline()
        # [YAW] [PITCH] [ROLL]
        eulers_txt = txt_line.strip().split(" ")[:3]
        kp_list = [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
        box_cur = txt_line.strip().split(" ")[3:]
        bndbox = []
        for index in range(len(box_cur) // 2):
            bndbox.append([box_cur[index * 2], box_cur[index * 2 + 1]])
        kp_id = -1
        for box in bndbox:
            kp_id = kp_id + 1
            x_coord = float(box[0])
            y_coord = float(box[1])
            if x_coord < 0 or y_coord < 0:
                continue
            kp_list[kp_id][0] = int(float(x_coord) / x_length * 96)
            kp_list[kp_id][1] = int(float(y_coord) / y_length * 96)
        return eulers_txt, kp_list, True
    return None, None, False


def get_infer_info(eulers, heatmap):
    """
    Summary.

    calculate inferred values from model output

    Args:
        eulers(np.ndarray): inferred YAW,PITCH,ROLL.
        heatmap(np.ndarray): inferred heatmap result.

    Returns:
        list, inferred YAW,PITCH,ROLL value and 5 key point coordinates..
    """
    kp_coord_ori = list()
    for i, _ in enumerate(heatmap):
        map_1 = heatmap[i].reshape(1, 48 * 48)
        # softmax
        map_1 = np.exp(map_1) / np.sum(np.exp(map_1), axis=1)
        kp_coor = map_1.argmax()
        kp_coor = int((kp_coor % 48) * 2.0), int((kp_coor / 48) * 2.0)
        kp_coord_ori.append(kp_coor)
    return kp_coord_ori, eulers


def save_output(out, output_dir, img_path):
    """
    Summary.

    Save output to files.

    Args:
        out(list): items need to be saved.
        output_dir(string): saved directory.
        img_path(string): image path.
    """
    if not output_dir:
        return None
    file_name = img_path.strip().split(os.path.sep)[-1].split('.')[0]
    infer_image_path = os.path.join(output_dir, f"{file_name}_infer.txt")
    output = open(infer_image_path, 'w')
    for item in out:
        output.write(str(item))
    return 1


def main(args_):
    kp_error_all = [[], [], [], [], []]
    eulers_error_all = [[], [], []]
    kp_ipn = []
    path = args_.dataset

    stream_manager_api = get_stream_manager(args_.pipeline)
    for img_path in get_dataset(path):
        out = []
        out_eul, heatmap = do_infer_image(stream_manager_api, img_path)
        eulers_gt, kp_list, euler_kps_do = read_ground_truth(img_path)
        if not euler_kps_do:
            continue

        kp_coord_ori, eulers_ori = get_infer_info(out_eul, heatmap)
        eulgt = list(eulers_gt)
        eulers_error = "eulers_error:"
        for euler_id, _ in enumerate(eulers_ori):
            eulers_error_all[euler_id].append(abs(eulers_ori[euler_id] - float(eulgt[euler_id])))
            eulers_error += " " + str(abs(eulers_ori[euler_id] - float(eulgt[euler_id])))
        out.append(eulers_error + "\n")
        eye01 = kp_list[0]
        eye02 = kp_list[1]
        eye_dis = 1
        cur_flag = True
        if eye01[0] < 0 or eye01[1] < 0 or eye02[0] < 0 or eye02[1] < 0:
            cur_flag = False
        else:
            eye_dis = np.sqrt(np.square(abs(eye01[0] - eye02[0])) + np.square(abs(eye01[1] - eye02[1])))
        cur_error_list = []
        for i in range(5):
            if kp_list[i][0] != -1:
                dis = np.sqrt(
                    np.square(kp_list[i][0] - kp_coord_ori[i][0]) + np.square(kp_list[i][1] - kp_coord_ori[i][1]))
                kp_error_all[i].append(dis)
                cur_error_list.append(dis)
        out.append("keypoint_error: " + str(cur_error_list) + "\n")
        save_output(out, args_.output, img_path)
        if cur_flag:
            kp_ipn.append(sum(cur_error_list) / len(cur_error_list) / eye_dis)

    kp_ave_error = []
    for kps, _ in enumerate(kp_error_all):
        kp_ave_error.append("%.3f" % (sum(kp_error_all[kps]) / len(kp_error_all[kps])))

    euler_ave_error = []
    elur_mae = []
    for eulers, _ in enumerate(eulers_error_all):
        euler_ave_error.append("%.3f" % (sum(eulers_error_all[eulers]) / len(eulers_error_all[eulers])))
        elur_mae.append((sum(eulers_error_all[eulers]) / len(eulers_error_all[eulers])))

    print('========== 5 keypoints average err:' + str(kp_ave_error))
    print('========== 3 eulers average err:' + str(euler_ave_error))
    print('========== IPN of 5 keypoints:' + str(sum(kp_ipn) / len(kp_ipn) * 100))
    print('========== MAE of elur:' + str(sum(elur_mae) / len(elur_mae)))
    stream_manager_api.DestroyAllStreams()
    return 1


if __name__ == "__main__":
    args = parse_arg()
    main(args)
