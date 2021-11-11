/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Cgan.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input data path, such as '../data'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.modelPath = "../data/model/CGAN.om";
    initParam.savePath = "../data/mxbase_result";
    auto cgan = std::make_shared<Cgan>();
    APP_ERROR ret = cgan->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Cgan init failed, ret=" << ret << ".";
        return ret;
    }

    std::string dataPath = argv[1];

    int imgNum = 1;
    int input_dim = 100;
    int n_image = 200;
    int n_col = 20;
    auto startTime = std::chrono::high_resolution_clock::now();

    cv::Mat latent_code_eval = cv::Mat(n_image, input_dim, CV_32FC1);
    cv::randn(latent_code_eval, 0, 1);

    cv::Mat label_eval = cv::Mat::zeros(n_image, 10, CV_32FC1);
    for (int i = 0; i < n_image; i++) {
        int j = i / n_col;
        label_eval.at<float>(i, j) = 1;
    }
    for (int i = 0; i < n_image; i++) {
        std::string imgName = std::to_string(i) + ".png";
        cv::Mat image;
        cv::Mat label;
        image = latent_code_eval.rowRange(i, i+1).clone();
        label = label_eval.rowRange(i, i+1).clone();
        ret = cgan->Process(image, label, imgName);
        if (ret != APP_ERR_OK) {
            LogError << "Cgan process failed, ret=" << ret << ".";
            cgan->DeInit();
            return ret;
        }
    }

    std::vector<cv::Mat> results;
    std::vector<cv::Mat> images;
    for (int i = 0; i < n_image; i++) {
        std::string filepath = dataPath + "/mxbase_result/" + std::to_string(i) + ".png";
        cv::Mat image = cv::imread(filepath);
        images.push_back(image);

        if ((i + 1) % n_col == 0) {
            cv::Mat result;
            cv::hconcat(images, result);
            LogInfo << "result" << result.size();
            images.clear();
            results.push_back(result);
        }

        if (i + 1 == n_image) {
            cv::Mat output;
            cv::vconcat(results, output);
            std::string resultpath = "../data/mxbase_result/result.png";
            cv::imwrite(resultpath, output);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    cgan->DeInit();
    double costMilliSecs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgNum / cgan->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps << " imgs/sec";
    return APP_ERR_OK;
}
