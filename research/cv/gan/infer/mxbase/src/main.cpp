/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <sys/io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <fstream>
#include "MxBase/Log/Log.h"
#include "GAN.h"

int main(int argc, char* argv[]) {
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/model/gan.om";

    auto model_gan = std::make_shared<gan>();
    APP_ERROR ret = model_gan->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Tagging init failed, ret=" << ret << ".";
        return ret;
    }

    for (int i = 0; i < 10000; i++) {
        std::string data_path, out_path;
        if (argv[1]) {
            data_path = std::string(argv[1]) + "/input_latent" + std::to_string(i) + ".bin";
            std::string s = "mkdir -p " + std::string(argv[2]);
            if (access(argv[2], 0) == -1) {
                system(s.c_str());
            }
            out_path = std::string(argv[2]) + "/" + std::to_string(i) + ".bin";
        } else {
            data_path = "../data/input/input_latent" + std::to_string(i) + ".bin";
            std::string s = "mkdir -p ../results/mxbase/";
            if (access(out_path.c_str(), 0) == -1) {
                system(s.c_str());
            }
            out_path = "../results/mxbase/" + std::to_string(i) + ".bin";
        }
        ret = model_gan->Process(data_path, initParam, out_path);
        if (ret !=APP_ERR_OK) {
            LogError << "Gan process failed, ret=" << ret << ".";
            model_gan->DeInit();
            return ret;
        }
    }
    model_gan->DeInit();

    double total_time = model_gan->GetInferCostMilliSec() / 1000;
    LogInfo<< "inferance total cost time: "<< total_time;
    return APP_ERR_OK;
}
