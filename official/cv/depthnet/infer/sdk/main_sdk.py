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

import os
import datetime
import argparse
import cv2
import numpy as np
from PIL import Image

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn


class TestDatasetGenerator:
    def __init__(self, test_data_dir):
        self.rgb_name_lists, self.depth_name_lists = self.get_rgb_and_depth_name_lists(test_data_dir)

    def __getitem__(self, index):
        rgb_name = self.rgb_name_lists[index]
        depth_name = self.depth_name_lists[index]
        rgb = Image.open(rgb_name)
        depth = Image.open(depth_name)
        rgb = np.array(rgb) / 255.0
        depth = np.array(depth) / 1000.0
        depth = depth.reshape((depth.shape[0], depth.shape[1], 1))
        return rgb, depth

    def __len__(self):
        return len(self.rgb_name_lists)

    def get_rgb_and_depth_name_lists(self, test_data_dir):
        rgb_name_lists = []
        depth_name_list = []
        file_lists = os.listdir(test_data_dir)
        file_lists.sort()
        for file_name in file_lists:
            if file_name[6:-4] == "colors":
                file_name = test_data_dir + "/" + file_name
                rgb_name_lists.append(file_name)
            if file_name[6:-4] == "depth":
                file_name = test_data_dir + "/" + file_name
                depth_name_list.append(file_name)

        return rgb_name_lists, depth_name_list


def threshold_percentage_loss(pred_depth, ground_truth, threshold_val):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)

    ground_truth = np.clip(ground_truth, 0.1, 10.0)

    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10.0)

    d1 = pred_depth / ground_truth
    d2 = ground_truth / pred_depth

    max_d1_d2 = np.maximum(d1, d2)
    bit_map = np.where(max_d1_d2 <= threshold_val, 1.0, 0.0)
    delta = bit_map.sum() / (bit_map.shape[0] * bit_map.shape[1])

    return delta


def rmse_linear(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)

    ground_truth = np.clip(ground_truth, 0.1, 10.0)

    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10.0)

    diff = pred_depth - ground_truth
    diff2 = np.power(diff, 2)
    mse = np.sum(diff2) / (pred_depth.shape[0] * pred_depth.shape[1])
    rmse = np.sqrt(mse)

    return rmse


def rmse_log(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)

    ground_truth = np.clip(ground_truth, 0.1, 10.0)

    diff = pred_depth - np.log(ground_truth)
    diff2 = np.power(diff, 2)
    mse = np.sum(diff2) / (pred_depth.shape[0] * pred_depth.shape[1])
    rmse = np.sqrt(mse)

    return rmse


def rmse_log_inv(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)
    ground_truth = np.clip(ground_truth, 0.1, 10.0)

    alpha = np.sum((np.log(ground_truth) - pred_depth)) / (pred_depth.shape[0] * pred_depth.shape[1])
    D = np.sum(np.power((pred_depth - np.log(ground_truth) + alpha), 2)) / (pred_depth.shape[0] * pred_depth.shape[1])

    return D


def abs_relative_difference(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)

    ground_truth = np.clip(ground_truth, 0.1, 10)

    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10)

    abs_relative_diff = np.abs(pred_depth - ground_truth) / ground_truth
    abs_relative_diff = np.sum(abs_relative_diff) / (pred_depth.shape[0] * pred_depth.shape[1])

    return abs_relative_diff


def squared_relative_difference(pred_depth, ground_truth):
    pred_depth = np.squeeze(pred_depth)
    ground_truth = np.squeeze(ground_truth)

    ground_truth = np.clip(ground_truth, 0.1, 10)

    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10)

    square_relative_diff = np.power(np.abs(pred_depth - ground_truth), 2) / ground_truth
    square_relative_diff = np.sum(square_relative_diff) / (pred_depth.shape[0] * pred_depth.shape[1])
    return square_relative_diff


def preprocess(cfg):
    dataset = TestDatasetGenerator(cfg.data_path)
    rgb_result_list = []
    depth_result_list = []
    rgb_list, depth_list = dataset.get_rgb_and_depth_name_lists(cfg.data_path)
    for rgb, depth in zip(rgb_list, depth_list):
        # the format of depth_png is single channel and 16 bits
        rgb_img = cv2.imread(rgb, cv2.IMREAD_COLOR)[:, :, ::-1]
        depth_png = cv2.imread(depth, cv2.CV_16UC1)
        rgb_img = rgb_img[12:468, 16:624]
        depth_png = depth_png[12:468, 16:624]
        rgb_final = cv2.resize(rgb_img, (304, 228), interpolation=cv2.INTER_LINEAR)
        depth_final = cv2.resize(depth_png, (74, 55), interpolation=cv2.INTER_LINEAR)
        rgb_final = np.float32(rgb_final)
        depth_final = np.float32(depth_final)
        rgb_final = np.transpose(rgb_final, axes=(2, 0, 1))
        # normalize and unit conversion
        rgb_final = rgb_final / 255.0
        depth_final = depth_final / 1000.0
        rgb_final = rgb_final.reshape((1, 3, 228, 304))
        depth_final = depth_final.reshape((1, 1, 55, 74))
        # save results
        rgb_result_list.append(rgb_final)
        depth_result_list.append(depth_final)
    return rgb_result_list, depth_result_list


def sdk_coarse_infer(cfg):
    # init streams
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(cfg.Coarse_PL_PATH, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    coarse_infer_list = []
    rgb_list, _ = preprocess(cfg)
    start_time = datetime.datetime.now()
    for rgb in rgb_list:
        stream_name = b'im_DepthNetCoarse'

        # send coarse data
        coarse_plugin_id = 0
        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionInfo.format = 0
        vision_vec.visionInfo.width = 304
        vision_vec.visionInfo.height = 228
        vision_vec.visionInfo.widthAligned = 304
        vision_vec.visionInfo.heightAligned = 228

        vision_vec.visionData.memType = 0
        vision_vec.visionData.dataStr = rgb.tobytes()
        vision_vec.visionData.dataSize = len(rgb)

        protobuf = MxProtobufIn()
        protobuf.key = b"appsrc0"
        protobuf.type = b'MxTools.MxpiVisionList'
        protobuf.protobuf = vision_list.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)

        coarse_unique_id = stream_manager_api.SendProtobuf(stream_name, 0, protobuf_vec)
        if coarse_unique_id < 0:
            print("Failed to send coarse data to stream.")
            exit()

        # do coarse infer
        coarse_keys = [b"mxpi_tensorinfer0"]
        Coarse_keyVec = StringVector()
        for key in coarse_keys:
            Coarse_keyVec.push_back(key)
        coarse_infer_result = stream_manager_api.GetProtobuf(stream_name, coarse_plugin_id, Coarse_keyVec)
        if coarse_infer_result.size() == 0:
            print("coarse_infer_result is null")
            exit()
        if coarse_infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                coarse_infer_result[0].errorCode, coarse_infer_result[0].data.decode()))
            exit()
        coarse_resultList = MxpiDataType.MxpiTensorPackageList()
        coarse_resultList.ParseFromString(coarse_infer_result[0].messageBuf)
        coarse_output = np.frombuffer(coarse_resultList.tensorPackageVec[0].tensorVec[0].dataStr,
                                      dtype='<f4').reshape((1, 1, 55, 74))
        # numpy->list
        coarse_infer_list.append(coarse_output)
    end_time = datetime.datetime.now()
    print('coarse infer run time: {}'.format((end_time - start_time).microseconds))
    return coarse_infer_list


def sdk_fine_infer(cfg):
    # init streams
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(cfg.Fine_PL_PATH, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    coarse_infer_list = sdk_coarse_infer(cfg)
    rgb_list, _ = preprocess(cfg)
    start_time = datetime.datetime.now()
    fine_infer_list = []
    for rgb, coarse_depth in zip(rgb_list, coarse_infer_list):
        stream_name = b'im_DepthNetFine'
        fine_plugin_id = 0

        # send rgb data
        vision_list_rgb = MxpiDataType.MxpiVisionList()
        vision_vec_rgb = vision_list_rgb.visionVec.add()
        vision_vec_rgb.visionInfo.format = 0
        vision_vec_rgb.visionInfo.width = 304
        vision_vec_rgb.visionInfo.height = 228
        vision_vec_rgb.visionInfo.widthAligned = 304
        vision_vec_rgb.visionInfo.heightAligned = 228

        vision_vec_rgb.visionData.memType = 0
        vision_vec_rgb.visionData.dataStr = rgb.tobytes()
        vision_vec_rgb.visionData.dataSize = len(rgb)

        protobuf_rgb = MxProtobufIn()
        protobuf_rgb.key = b"appsrc0"
        protobuf_rgb.type = b'MxTools.MxpiVisionList'
        protobuf_rgb.protobuf = vision_list_rgb.SerializeToString()
        protobuf_vec_rgb = InProtobufVector()
        protobuf_vec_rgb.push_back(protobuf_rgb)

        rgb_unique_id = stream_manager_api.SendProtobuf(stream_name, 0, protobuf_vec_rgb)
        if rgb_unique_id < 0:
            print("Failed to send rgb data to stream.")
            exit()

        # send depth data
        vision_list_depth = MxpiDataType.MxpiVisionList()
        vision_vec_depth = vision_list_depth.visionVec.add()
        vision_vec_depth.visionInfo.format = 0
        vision_vec_depth.visionInfo.width = 74
        vision_vec_depth.visionInfo.height = 55
        vision_vec_depth.visionInfo.widthAligned = 74
        vision_vec_depth.visionInfo.heightAligned = 55

        vision_vec_depth.visionData.memType = 0
        vision_vec_depth.visionData.dataStr = coarse_depth.tobytes()
        vision_vec_depth.visionData.dataSize = len(coarse_depth)

        protobuf_depth = MxProtobufIn()
        protobuf_depth.key = b"appsrc1"
        protobuf_depth.type = b'MxTools.MxpiVisionList'
        protobuf_depth.protobuf = vision_list_depth.SerializeToString()
        protobuf_vec_depth = InProtobufVector()
        protobuf_vec_depth.push_back(protobuf_depth)

        depth_unique_id = stream_manager_api.SendProtobuf(stream_name, 1, protobuf_vec_depth)
        if depth_unique_id < 0:
            print("Failed to send depth data to stream.")
            exit()

        # do fine infer
        fine_keys = [b"mxpi_tensorinfer0"]
        Fine_keyVec = StringVector()
        for key in fine_keys:
            Fine_keyVec.push_back(key)
        fine_infer_result = stream_manager_api.GetProtobuf(stream_name, fine_plugin_id, Fine_keyVec)
        if fine_infer_result.size() == 0:
            print("fine_infer_result is null")
            exit()
        if fine_infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                fine_infer_result[0].errorCode, fine_infer_result[0].data.decode()))
            exit()
        fine_resultList = MxpiDataType.MxpiTensorPackageList()
        fine_resultList.ParseFromString(fine_infer_result[0].messageBuf)
        fine_output = np.frombuffer(fine_resultList.tensorPackageVec[0].tensorVec[0].dataStr,
                                    dtype='<f4').reshape((1, 1, 55, 74))
        # numpy->list
        fine_infer_list.append(fine_output)
    end_time = datetime.datetime.now()
    print('fine infer run time: {}'.format((end_time - start_time).microseconds))
    return fine_infer_list


def save_infer_result(cfg):
    dataset = TestDatasetGenerator(cfg.data_path)
    coarse_path = cfg.coarse_infer_result_url
    if not os.path.exists(coarse_path):
        os.mkdir(coarse_path)
    fine_path = cfg.fine_infer_result_url
    if not os.path.exists(fine_path):
        os.mkdir(fine_path)
    rgb_list, depth_list = dataset.get_rgb_and_depth_name_lists(cfg.data_path)
    coarse_list = sdk_coarse_infer(cfg)
    fine_list = sdk_fine_infer(cfg)
    for rgb, depth, coarse, fine in zip(rgb_list, depth_list, coarse_list, fine_list):
        # save coarse_infer_result
        coarse_path = rgb.replace(cfg.data_path, cfg.coarse_infer_result_url)
        if coarse_path.find("colors"):
            coarse_path = coarse_path.replace("colors.png", "coarse_depth.bin")
        coarse.tofile(coarse_path)

        # save fine_infer_result
        fine_path = depth.replace(cfg.data_path, cfg.fine_infer_result_url)
        if fine_path.find("depth"):
            fine_path = fine_path.replace("depth.png", "fine_depth.bin")
        fine.tofile(fine_path)
    return coarse_list, fine_list


def sdk_eval(cfg):
    # eval args init
    delta1 = 1.25
    delta2 = 1.25 ** 2
    delta3 = 1.25 ** 3

    coarse_total_delta1 = 0
    coarse_total_delta2 = 0
    coarse_total_delta3 = 0
    coarse_total_abs_relative_loss = 0
    coarse_total_sqr_relative_loss = 0
    coarse_total_rmse_linear_loss = 0
    coarse_total_rmse_log_loss = 0
    coarse_total_rmse_log_inv_loss = 0

    fine_total_delta1 = 0
    fine_total_delta2 = 0
    fine_total_delta3 = 0
    fine_total_abs_relative_loss = 0
    fine_total_sqr_relative_loss = 0
    fine_total_rmse_linear_loss = 0
    fine_total_rmse_log_loss = 0
    fine_total_rmse_log_inv_loss = 0

    index = 0

    # get ground_truth and sdk_infer_result
    _, gt_list = preprocess(cfg)
    coarse_list, fine_list = save_infer_result(cfg)

    for coarse_depth, fine_depth, ground_truth in zip(coarse_list, fine_list, gt_list):
        coarse_delta1_loss = threshold_percentage_loss(coarse_depth, ground_truth, delta1)
        coarse_delta2_loss = threshold_percentage_loss(coarse_depth, ground_truth, delta2)
        coarse_delta3_loss = threshold_percentage_loss(coarse_depth, ground_truth, delta3)

        coarse_abs_relative_loss = abs_relative_difference(coarse_depth, ground_truth)
        coarse_sqr_relative_loss = squared_relative_difference(coarse_depth, ground_truth)
        coarse_rmse_linear_loss = rmse_linear(coarse_depth, ground_truth)
        coarse_rmse_log_loss = rmse_log(coarse_depth, ground_truth)
        coarse_rmse_log_inv_loss = rmse_log_inv(coarse_depth, ground_truth)

        fine_delta1_loss = threshold_percentage_loss(fine_depth, ground_truth, delta1)
        fine_delta2_loss = threshold_percentage_loss(fine_depth, ground_truth, delta2)
        fine_delta3_loss = threshold_percentage_loss(fine_depth, ground_truth, delta3)

        fine_abs_relative_loss = abs_relative_difference(fine_depth, ground_truth)
        fine_sqr_relative_loss = squared_relative_difference(fine_depth, ground_truth)
        fine_rmse_linear_loss = rmse_linear(fine_depth, ground_truth)
        fine_rmse_log_loss = rmse_log(fine_depth, ground_truth)
        fine_rmse_log_inv_loss = rmse_log_inv(fine_depth, ground_truth)

        coarse_total_delta1 += coarse_delta1_loss
        coarse_total_delta2 += coarse_delta2_loss
        coarse_total_delta3 += coarse_delta3_loss
        coarse_total_abs_relative_loss += coarse_abs_relative_loss
        coarse_total_sqr_relative_loss += coarse_sqr_relative_loss
        coarse_total_rmse_linear_loss += coarse_rmse_linear_loss
        coarse_total_rmse_log_loss += coarse_rmse_log_loss
        coarse_total_rmse_log_inv_loss += coarse_rmse_log_inv_loss

        fine_total_delta1 += fine_delta1_loss
        fine_total_delta2 += fine_delta2_loss
        fine_total_delta3 += fine_delta3_loss
        fine_total_abs_relative_loss += fine_abs_relative_loss
        fine_total_sqr_relative_loss += fine_sqr_relative_loss
        fine_total_rmse_linear_loss += fine_rmse_linear_loss
        fine_total_rmse_log_loss += fine_rmse_log_loss
        fine_total_rmse_log_inv_loss += fine_rmse_log_inv_loss

        print("test case ", index)
        print("CoarseNet: \n", "delta1_loss: ", coarse_delta1_loss, " delta2_loss: ", coarse_delta2_loss,
              " delta3_loss: ", coarse_delta3_loss, " abs_relative_loss: ", coarse_abs_relative_loss,
              " sqr_relative_loss: ", coarse_sqr_relative_loss, " rmse_linear_loss: ", coarse_rmse_linear_loss,
              " rmse_log_loss: ", coarse_rmse_log_loss, " rmse_log_inv_loss: ", coarse_rmse_log_inv_loss)

        print("FineNet: \n",
              "delta1_loss: ", fine_delta1_loss, " delta2_loss: ", fine_delta2_loss, " delta3_loss: ", fine_delta3_loss,
              " abs_relative_loss: ", fine_abs_relative_loss, " sqr_relative_loss: ", fine_sqr_relative_loss,
              " rmse_linear_loss: ", fine_rmse_linear_loss, " rmse_log_loss: ", fine_rmse_log_loss,
              " rmse_log_inv_loss: ", fine_rmse_log_inv_loss)

        print("\n")
        index += 1

    coarse_average_delta1 = coarse_total_delta1 / index
    coarse_average_delta2 = coarse_total_delta2 / index
    coarse_average_delta3 = coarse_total_delta3 / index

    coarse_average_abs_relative_loss = coarse_total_abs_relative_loss / index
    coarse_average_sqr_relative_loss = coarse_total_sqr_relative_loss / index
    coarse_average_rmse_linear_loss = coarse_total_rmse_linear_loss / index
    coarse_average_rmse_log_loss = coarse_total_rmse_log_loss / index
    coarse_average_rmse_log_inv_loss = coarse_total_rmse_log_inv_loss / index

    fine_average_delta1 = fine_total_delta1 / index
    fine_average_delta2 = fine_total_delta2 / index
    fine_average_delta3 = fine_total_delta3 / index

    fine_average_abs_relative_loss = fine_total_abs_relative_loss / index
    fine_average_sqr_relative_loss = fine_total_sqr_relative_loss / index
    fine_average_rmse_linear_loss = fine_total_rmse_linear_loss / index
    fine_average_rmse_log_loss = fine_total_rmse_log_loss / index
    fine_average_rmse_log_inv_loss = fine_total_rmse_log_inv_loss / index

    print("average test accuracy:")
    print("CoarseNet: \n",
          "delta1_loss: ", coarse_average_delta1, " delta2_loss: ", coarse_average_delta2,
          " delta3_loss: ", coarse_average_delta3, " abs_relative_loss: ", coarse_average_abs_relative_loss,
          " sqr_relative_loss: ", coarse_average_sqr_relative_loss,
          " rmse_linear_loss: ", coarse_average_rmse_linear_loss,
          " rmse_log_loss: ", coarse_average_rmse_log_loss, " rmse_log_inv_loss: ", coarse_average_rmse_log_inv_loss)

    print("FineNet: \n",
          "delta1_loss: ", fine_average_delta1, " delta2_loss: ", fine_average_delta2,
          " delta3_loss: ", fine_average_delta3, " abs_relative_loss: ", fine_average_abs_relative_loss,
          " sqr_relative_loss: ", fine_average_sqr_relative_loss, " rmse_linear_loss: ", fine_average_rmse_linear_loss,
          " rmse_log_loss: ", fine_average_rmse_log_loss, " rmse_log_inv_loss: ", fine_average_rmse_log_inv_loss)


def parse_args():
    parser = argparse.ArgumentParser(description='DepthNet Inferring sdk')
    # Datasets
    parser.add_argument('--data_path', default='../input/data/nyu2_test', type=str,
                        help='test data path')
    parser.add_argument('--Coarse_PL_PATH', default='./config/CoarseNet.pipeline', type=str,
                        help='coarse model pipeline path')
    parser.add_argument('--Fine_PL_PATH', default='./config/FineNet.pipeline', type=str,
                        help='fine model pipeline path')
    parser.add_argument('--coarse_infer_result_url', default='./coarse_infer_result', type=str,
                        help='coarse model infer result path')
    parser.add_argument('--fine_infer_result_url', default='./fine_infer_result', type=str,
                        help='fine model infer result path')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    sdk_eval(cfg=args)
