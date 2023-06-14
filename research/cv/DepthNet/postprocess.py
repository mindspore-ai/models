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
"""post process for 310 inference"""
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="postprocess for googlenet")
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--label_file", type=str, required=True, help="label file")
args = parser.parse_args()


def threshold_percentage_loss(pred_depth, ground_truth, threshold_val):
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
    ground_truth = np.clip(ground_truth, 0.1, 10.0)
    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10.0)
    diff = pred_depth - ground_truth
    diff2 = np.power(diff, 2)
    mse = np.sum(diff2) / (pred_depth.shape[0] * pred_depth.shape[1])
    rmse = np.sqrt(mse)
    return rmse


def rmse_log(pred_depth, ground_truth):
    ground_truth = np.clip(ground_truth, 0.1, 10.0)

    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10.0)

    diff = np.log(pred_depth) - np.log(ground_truth)
    diff2 = np.power(diff, 2)
    mse = np.sum(diff2) / (pred_depth.shape[0] * pred_depth.shape[1])
    rmse = np.sqrt(mse)
    return rmse


def abs_relative_difference(pred_depth, ground_truth):
    ground_truth = np.clip(ground_truth, 0.1, 10)
    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10)
    abs_relative_diff = np.abs(pred_depth - ground_truth) / ground_truth
    abs_relative_diff = np.sum(abs_relative_diff) / (pred_depth.shape[0] * pred_depth.shape[1])
    return abs_relative_diff


def squared_relative_difference(pred_depth, ground_truth):
    ground_truth = np.clip(ground_truth, 0.1, 10)
    pred_depth = np.exp(pred_depth)
    pred_depth = np.clip(pred_depth, 0.1, 10)
    square_relative_diff = np.power(np.abs(pred_depth - ground_truth), 2) / ground_truth
    square_relative_diff = np.sum(square_relative_diff) / (pred_depth.shape[0] * pred_depth.shape[1])
    return square_relative_diff


delta1 = 1.25
delta2 = 1.25**2
delta3 = 1.25**3


def cal_acc_nyu(result_path, label_path):
    result_shape = (55, 74)
    files = os.listdir(result_path)
    delta1_ = []
    delta2_ = []
    delta3_ = []
    abs_relative_ = []
    sqr_relative_ = []
    rmse_linear_ = []
    rmse_log_ = []
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(result_shape)
            label_file = os.path.join(label_path, file.split(".bin")[0][:-2] + ".bin")
            ground_truth = np.fromfile(label_file, dtype=np.float32).reshape(result_shape)

            delta1_loss = threshold_percentage_loss(result, ground_truth, delta1)
            delta2_loss = threshold_percentage_loss(result, ground_truth, delta2)
            delta3_loss = threshold_percentage_loss(result, ground_truth, delta3)

            abs_relative_loss = abs_relative_difference(result, ground_truth)
            sqr_relative_loss = squared_relative_difference(result, ground_truth)
            rmse_linear_loss = rmse_linear(result, ground_truth)
            rmse_log_loss = rmse_log(result, ground_truth)

            delta1_.append(delta1_loss)
            delta2_.append(delta2_loss)
            delta3_.append(delta3_loss)
            abs_relative_.append(abs_relative_loss)
            sqr_relative_.append(sqr_relative_loss)
            rmse_linear_.append(rmse_linear_loss)
            rmse_log_.append(rmse_log_loss)

    delta1_ave = sum(delta1_) / len(delta1_)
    delta2_ave = sum(delta2_) / len(delta2_)
    delta3_ave = sum(delta3_) / len(delta3_)
    abs_relative_ave = sum(abs_relative_) / len(abs_relative_)
    sqr_relative_ave = sum(sqr_relative_) / len(sqr_relative_)
    rmse_linear_ave = sum(rmse_linear_) / len(rmse_linear_)
    rmse_log_ave = sum(rmse_log_) / len(rmse_log_)
    print("average test accuracy:")
    print(
        "delta1_loss: ",
        delta1_ave,
        " delta2_loss: ",
        delta2_ave,
        " delta3_loss: ",
        delta3_ave,
        " abs_relative_loss: ",
        abs_relative_ave,
        " sqr_relative_loss: ",
        sqr_relative_ave,
        " rmse_linear_loss: ",
        rmse_linear_ave,
        " rmse_log_loss: ",
        rmse_log_ave,
    )


if __name__ == "__main__":
    cal_acc_nyu(args.result_path, args.label_file)
