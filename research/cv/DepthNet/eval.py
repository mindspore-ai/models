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

import os
import argparse

from mindspore import Tensor, Model
from mindspore import load_checkpoint, load_param_into_net

from src.net import CoarseNet
from src.net import FineNet
from src.loss import threshold_percentage_loss
from src.loss import rmse_linear, rmse_log, rmse_log_inv
from src.loss import abs_relative_difference, squared_relative_difference
from src.data_loader import create_test_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindSpore Depth Estimation Demo")
    parser.add_argument("--test_data", type=str, default="./DepthNet_dataset", help="train data path")
    parser.add_argument(
        "--coarse_ckpt_model", type=str, default="./Model/Ckpt/FinalCoarseNet.ckpt", help="coarse model ckpt path"
    )
    parser.add_argument(
        "--fine_ckpt_model", type=str, default="./Model/Ckpt/FinalFineNet.ckpt", help="fine model ckpt path"
    )
    args = parser.parse_args()

    test_data_dir = os.path.join(args.test_data, "Test")
    coarse_ckpt_model = args.coarse_ckpt_model
    fine_ckpt_model = args.fine_ckpt_model
    test_dataset = create_test_dataset(test_data_dir, batch_size=1)

    coarse_net = CoarseNet()
    coarse_param_dict = load_checkpoint(coarse_ckpt_model)
    load_param_into_net(coarse_net, coarse_param_dict)
    coarse_net.set_train(mode=False)
    coarse_model = Model(coarse_net)

    fine_net = FineNet()
    fine_param_dict = load_checkpoint(fine_ckpt_model)
    load_param_into_net(fine_net, fine_param_dict)
    fine_net.set_train(mode=False)
    fine_model = Model(fine_net)

    delta1 = 1.25
    delta2 = 1.25**2
    delta3 = 1.25**3

    coarse_total_delta1 = 0
    coarse_total_delta2 = 0
    coarse_total_delta3 = 0
    coarse_total_abs_relative_loss = 0
    coarse_total_sqr_relative_loss = 0
    coarse_total_rmse_linear_loss = 0
    coarse_total_rmse_log_loss = 0
    coarse_total_train_loss = 0
    coarse_total_rmse_log_inv_loss = 0

    fine_total_delta1 = 0
    fine_total_delta2 = 0
    fine_total_delta3 = 0
    fine_total_abs_relative_loss = 0
    fine_total_sqr_relative_loss = 0
    fine_total_rmse_linear_loss = 0
    fine_total_rmse_log_loss = 0
    fine_total_train_loss = 0
    fine_total_rmse_log_inv_loss = 0

    index = 0
    for data in test_dataset.create_dict_iterator():
        rgb = data["rgb"]
        ground_truth = data["ground_truth"]
        coarse_depth = coarse_model.predict(Tensor(rgb))
        fine_depth = fine_model.predict(Tensor(rgb), coarse_depth)

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
        print(
            "CoarseNet: \n",
            "delta1_loss: ",
            coarse_delta1_loss,
            " delta2_loss: ",
            coarse_delta2_loss,
            " delta3_loss: ",
            coarse_delta3_loss,
            " abs_relative_loss: ",
            coarse_abs_relative_loss,
            " sqr_relative_loss: ",
            coarse_sqr_relative_loss,
            " rmse_linear_loss: ",
            coarse_rmse_linear_loss,
            " rmse_log_loss: ",
            coarse_rmse_log_loss,
            " rmse_log_inv_loss: ",
            coarse_rmse_log_inv_loss,
        )

        print(
            "FineNet: \n",
            "delta1_loss: ",
            fine_delta1_loss,
            " delta2_loss: ",
            fine_delta2_loss,
            " delta3_loss: ",
            fine_delta3_loss,
            " abs_relative_loss: ",
            fine_abs_relative_loss,
            " sqr_relative_loss: ",
            fine_sqr_relative_loss,
            " rmse_linear_loss: ",
            fine_rmse_linear_loss,
            " rmse_log_loss: ",
            fine_rmse_log_loss,
            " rmse_log_inv_loss: ",
            fine_rmse_log_inv_loss,
        )
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
    print(
        "CoarseNet: \n",
        "delta1_loss: ",
        coarse_average_delta1,
        " delta2_loss: ",
        coarse_average_delta2,
        " delta3_loss: ",
        coarse_average_delta3,
        " abs_relative_loss: ",
        coarse_average_abs_relative_loss,
        " sqr_relative_loss: ",
        coarse_average_sqr_relative_loss,
        " rmse_linear_loss: ",
        coarse_average_rmse_linear_loss,
        " rmse_log_loss: ",
        coarse_average_rmse_log_loss,
        " rmse_log_inv_loss: ",
        coarse_average_rmse_log_inv_loss,
    )

    print(
        "FineNet: \n",
        "delta1_loss: ",
        fine_average_delta1,
        " delta2_loss: ",
        fine_average_delta2,
        " delta3_loss: ",
        fine_average_delta3,
        " abs_relative_loss: ",
        fine_average_abs_relative_loss,
        " sqr_relative_loss: ",
        fine_average_sqr_relative_loss,
        " rmse_linear_loss: ",
        fine_average_rmse_linear_loss,
        " rmse_log_loss: ",
        fine_average_rmse_log_loss,
        " rmse_log_inv_loss: ",
        fine_average_rmse_log_inv_loss,
    )
