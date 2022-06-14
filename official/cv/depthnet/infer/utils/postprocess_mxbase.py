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
import argparse
import numpy as np


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


def get_result_lists(cfg):
    gt_dir = cfg.ground_truth_path
    coarse_result_dir = cfg.coarse_infer_result_path
    fine_result_dir = cfg.fine_infer_result_path
    gt_result_list = []
    coarse_result_list = []
    fine_result_list = []
    gt_lists = os.listdir(gt_dir)
    gt_lists.sort()
    coarse_lists = os.listdir(coarse_result_dir)
    coarse_lists.sort()
    fine_lists = os.listdir(fine_result_dir)
    fine_lists.sort()
    # get ground_truth list
    for gt in gt_lists:
        if gt[6:-4] == "depth":
            gt_file_name = gt_dir + "/" + gt
            f1 = open(gt_file_name, "rb")
            strb1 = f1.read()
            gt_result = np.frombuffer(strb1, dtype='<f4').reshape((1, 1, 55, 74))
            gt_result_list.append(gt_result)

    # get coarse and fine infer result list
    for coarse_depth, fine_depth in zip(coarse_lists, fine_lists):
        coarse_file_name = coarse_result_dir + "/" + coarse_depth
        f2 = open(coarse_file_name, "rb")
        strb2 = f2.read()
        coarse_result = np.frombuffer(strb2, dtype='<f4').reshape((1, 1, 55, 74))
        coarse_result_list.append(coarse_result)
        fine_file_name = fine_result_dir + "/" + fine_depth
        f3 = open(fine_file_name, "rb")
        strb3 = f3.read()
        fine_result = np.frombuffer(strb3, dtype='<f4').reshape((1, 1, 55, 74))
        fine_result_list.append(fine_result)
    return gt_result_list, coarse_result_list, fine_result_list


def mxbase_eval(cfg):
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

    # get ground_truth and mxbase_infer_result
    gt_list, coarse_list, fine_list = get_result_lists(cfg)

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
    print("CoarseNet: \n", "delta1_loss: ", coarse_average_delta1, " delta2_loss: ", coarse_average_delta2,
          " delta3_loss: ", coarse_average_delta3, " abs_relative_loss: ", coarse_average_abs_relative_loss,
          " sqr_relative_loss: ", coarse_average_sqr_relative_loss,
          "rmse_linear_loss: ", coarse_average_rmse_linear_loss,
          " rmse_log_loss: ", coarse_average_rmse_log_loss, " rmse_log_inv_loss: ", coarse_average_rmse_log_inv_loss)

    print("FineNet: \n",
          "delta1_loss: ", fine_average_delta1, " delta2_loss: ", fine_average_delta2,
          " delta3_loss: ", fine_average_delta3, " abs_relative_loss: ", fine_average_abs_relative_loss,
          " sqr_relative_loss: ", fine_average_sqr_relative_loss, " rmse_linear_loss: ", fine_average_rmse_linear_loss,
          " rmse_log_loss: ", fine_average_rmse_log_loss, " rmse_log_inv_loss: ", fine_average_rmse_log_inv_loss)


def parse_args():
    parser = argparse.ArgumentParser(description='DepthNet mxbase eval')
    # Datasets
    parser.add_argument('--ground_truth_path', default='../mxbase/mxbase_out', type=str,
                        help='gt preprocessed result path')
    parser.add_argument('--coarse_infer_result_path', default='../mxbase/coarse_infer_result', type=str,
                        help='coarse model infer_result path')
    parser.add_argument('--fine_infer_result_path', default='../mxbase/fine_infer_result', type=str,
                        help='fine model infer_result path')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    mxbase_eval(cfg=args)
