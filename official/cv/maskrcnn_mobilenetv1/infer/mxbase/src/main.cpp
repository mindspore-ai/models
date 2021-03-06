/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. 
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

#include "MaskRCNN.h"
#include "MxBase/Log/Log.h"

namespace {
const uint32_t DEVICE_ID = 0;
const char RESULT_PATH[100] = "../data/predict_result.json";

// parameters of post process
const uint32_t CLASS_NUM = 80;
const float SCORE_THRESH = 0.7;
const float IOU_THRESH = 0.5;
const char LABEL_PATH[100] = "../data/config/coco2017_to_labels.names";
}  // namespace

int main(int argc, char *argv[]) {
    if (argc <= 2) {
        LogWarn << "Please input image path, such as './maskrcnn_mindspore [om_file_path] [img_path]'.";
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
    auto inferMaskrcnn = std::make_shared<MaskRCNN>();
    APP_ERROR ret = inferMaskrcnn->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "MaskRCNN init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[2];
    ret = inferMaskrcnn->Process(imgPath, RESULT_PATH);
    if (ret != APP_ERR_OK) {
        LogError << "MaskRCNN process failed, ret=" << ret << ".";
        inferMaskrcnn->DeInit();
        return ret;
    }
    inferMaskrcnn->DeInit();
    return APP_ERR_OK;
}
