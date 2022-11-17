/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <vector>

#include "MxBase/Log/Log.h"
#include "Naml.h"

void InitSourceParam(InitParam* initParam, const std::string &model_dir) {
    initParam->deviceId = 0;
    initParam->newsmodelPath = model_dir + "naml_news_encoder_bs_1.om";
    initParam->usermodelPath = model_dir + "naml_user_encoder_bs_1.om";
    initParam->newsDataPath = "../data/mxbase_data/newsdata.csv";
    initParam->userDataPath = "../data/mxbase_data/userdata.csv";
    initParam->evalDataPath = "../data/mxbase_data/evaldata.csv";
}

int main(int argc, char const* argv[]) {
    if (argc <= 1) {
        LogError << "Please input model dir , such as './data/model/";
        return APP_ERR_OK;
    }
    InitParam initParam;
    std::string m_path = argv[1];
    InitSourceParam(&initParam, m_path);
    auto namlbase = std::make_shared < Naml >();
    std::vector < std::string > datapaths;
    datapaths.push_back(initParam.newsDataPath);
    datapaths.push_back(initParam.userDataPath);
    datapaths.push_back(initParam.evalDataPath);

    APP_ERROR ret = namlbase->process(datapaths, initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Naml process failed, ret=" << ret << ".";
        return ret;
    }
    namlbase->de_init();
    return 0;
}
