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
#include <dirent.h>
#include <algorithm>
#include "FaceRecognitionForTracking.h"
#include "MxBase/Log/Log.h"

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './ghostnet /path/to/jpeg_image_dir'.";
        return APP_ERR_OK;
    }

    std::string imgPath = argv[1];
    InitParam initParam;
    initParam.deviceId = 0;
    initParam.modelPath = "../../data/model/face_recognition_for_tracking.om";
    auto face_recognition_for_tracking = std::make_shared<FaceRecognitionForTracking>();
    APP_ERROR ret = face_recognition_for_tracking->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FaceRecognitionForTracking init failed, ret=" << ret << ".";
        return ret;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    ret = face_recognition_for_tracking->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "FaceRecognitionForTracking process failed, ret=" << ret << ".";
        face_recognition_for_tracking->DeInit();
        return ret;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    face_recognition_for_tracking->DeInit();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    LogInfo << "[Total process delay] cost: " << costMs << " ms";
    return APP_ERR_OK;
}
