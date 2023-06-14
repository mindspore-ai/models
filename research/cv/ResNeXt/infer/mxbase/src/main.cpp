/*
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */
#include <experimental/filesystem>
#include "resnext50Classify.h"
#include "MxBase/Log/Log.h"

namespace fs = std::experimental::filesystem;
namespace {
const uint32_t CLASS_NUM = 1000;
}
std::vector<double> g_inferCost;

int main(int argc, char* argv[])
{   if (argc <= 1) {
        LogWarn << "Please input image path, such as './resnext ../data/input 10'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../data/model/imagenet1000_clsidx_to_labels.names";
    initParam.topk = 5;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/resnext50.om";
    auto resnext50 = std::make_shared<resnext50Classify>();
    APP_ERROR ret = resnext50->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "resnext50Classify init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgDir = argv[1];
    int limit = std::strtol(argv[2], nullptr, 0);
    int index = 0;
    for (auto & entry : fs::directory_iterator(imgDir)) {
        if (limit > 0 && index == limit) {
            break;
        }
        index++;
        LogInfo << "read image path " << entry.path();
        ret = resnext50->Process(entry.path());
        if (ret != APP_ERR_OK) {
            LogError << "resnext50Classify process failed, ret=" << ret << ".";
            resnext50->DeInit();
            return ret;
        }
    }
    resnext50->DeInit();
    double costSum = 0;
    for (unsigned int i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}
