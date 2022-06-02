/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd. 
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
 */

#include "MxBase/Log/Log.h"
#include "Neighbor2Neighbor.h"


int main(int argc, char *argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path, such as './ [om_file_path] [img_path] [dataset_name]'.";
        return APP_ERR_OK;
    }
    InitParam initParam = {};

    initParam.checkTensor = true;

    initParam.modelPath = argv[1];
    auto inferNeighbor2Neighbor = std::make_shared<Neighbor2Neighbor>();
    APP_ERROR ret = inferNeighbor2Neighbor->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Alphapose init failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgPath = argv[2];
    ret = inferNeighbor2Neighbor->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "Vehiclenet process failed, ret=" << ret << ".";
        inferNeighbor2Neighbor->DeInit();
        return ret;
    }
    inferNeighbor2Neighbor->DeInit();
    return APP_ERR_OK;
}
