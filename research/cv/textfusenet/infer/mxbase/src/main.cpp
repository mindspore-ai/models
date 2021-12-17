/*
 * Copyright 2021. Huawei Technologies Co., Ltd.
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

#include "Textfusenet.h"
#include <experimental/filesystem>
#include "MxBase/Log/Log.h"
namespace fs = std::experimental::filesystem;

namespace {
const uint32_t DEVICE_ID = 0;
// parameters of post process
const uint32_t CLASS_NUM = 63;
const float SCORE_THRESH = 0;
const float IOU_THRESH = 1;
const char LABEL_PATH[]  = "../data/models/textfusenet.names";

}  // namespace

int main(int argc, char *argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path, such as './textfusenet_mindspore [om_file_path] [img_path]'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = DEVICE_ID;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = LABEL_PATH;

    initParam.iou_thresh = IOU_THRESH;
    initParam.score_thresh = SCORE_THRESH;
    initParam.checkTensor = true;

    initParam.modelPath = argv[1];
    auto inferTextfusenet = std::make_shared<Textfusenet>();
    APP_ERROR ret = inferTextfusenet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Textfusenet init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[2];
    std::string outputPath = "./infer_result/";
    fs::create_directory(outputPath);
    for (auto & entry : fs::directory_iterator(imgPath)) {
        std::string src_path = entry.path();
        int pos_start = src_path.find_last_of('/');
        int pos_end = src_path.find_last_of('.');
        std::string file_name = src_path.substr(pos_start+1, pos_end-pos_start-1);
        std::string RESULT_PATH = outputPath + file_name + ".json";
        ret = inferTextfusenet->Process(src_path, RESULT_PATH);
        if (ret != APP_ERR_OK) {
            LogError << "Textfusenet process failed, ret=" << ret << ".";
            inferTextfusenet->DeInit();
            return ret;
        }
    }
    inferTextfusenet->DeInit();
    return APP_ERR_OK;
}
