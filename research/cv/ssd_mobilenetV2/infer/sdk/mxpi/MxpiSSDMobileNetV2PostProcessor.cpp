/*
 * Copyright(C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "infer/sdk/mxpi/MxpiSSDMobileNetV2PostProcessor.h"

#include <infer/mxbase/SSDPostProcessor.h>

#include <algorithm>
#include <memory>
#include <utility>

std::shared_ptr<MxpiSSDMobileNetV2PostProcess> GetInstance() {
    return std::make_shared<MxpiSSDMobileNetV2PostProcess>();
}

std::shared_ptr<sdk_infer::mxbase_infer::SSDPostProcessor> GetObjectInstance() {
    static std::shared_ptr<sdk_infer::mxbase_infer::SSDPostProcessor> instance =
        std::make_shared<sdk_infer::mxbase_infer::SSDPostProcessor>();
    LogInfo << "instance : " << instance.get();
    return instance;
}

APP_ERROR MxpiSSDMobileNetV2PostProcess::Init(const std::string &configPath,
                                              const std::string &labelPath,
                                              MxBase::ModelDesc modelDesc) {
    LogInfo << "Begin to init MxpiSSDMobileNetV2PostProcess";

    APP_ERROR ret = this->LoadConfigDataAndLabelMap(configPath, labelPath);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "LoadConfigDataAndLabelMap fail";
        return ret;
    }
    GetModelTensorsShape(modelDesc);
    if (checkModelFlag_) {
        ret = CheckModelCompatibility();
        if (ret != APP_ERR_OK) {
            LogError
                << GetError(ret)
                << "Fail to CheckModelCompatibility in ResNet50PostProcessor."
                << "Please check the compatibility between model and "
                   "postprocessor";
            return ret;
        }
    } else {
        LogWarn
            << "Compatibility check for model is skipped as CHECK_MODEL is set "
               "as false, please ensure your model"
            << "is correct before running.";
    }
    LogInfo << "End to initialize ResNet50PostProcessor.";
    return APP_ERR_OK;
}

APP_ERROR MxpiSSDMobileNetV2PostProcess::DeInit() {
    LogInfo << "Begin to deinitialize ResNet50PostProcessor.";
    LogInfo << "End to deinitialize ResNet50PostProcessor.";
    return APP_ERR_OK;
}

APP_ERROR MxpiSSDMobileNetV2PostProcess::CheckModelCompatibility() {
    if (outputTensorShapes_.size() != 2) {
        LogError << "outputTensorShapes_.size() != 2.";
        return APP_ERR_OUTPUT_NOT_MATCH;
    }
    return APP_ERR_OK;
}

APP_ERROR MxpiSSDMobileNetV2PostProcess::Process(
    std::shared_ptr<void> &metaDataPtr,
    MxBase::PostProcessorImageInfo postProcessorImageInfo,
    std::vector<MxTools::MxpiMetaHeader> &headerVec,
    std::vector<std::vector<MxBase::BaseTensor>> &tensors) {
    int num_class = outputTensorShapes_[1][2];
    int num_box = outputTensorShapes_[1][1];
    if (metaDataPtr == nullptr) {
        metaDataPtr = std::static_pointer_cast<void>(
            std::make_shared<MxTools::MxpiObjectList>());
    }
    std::shared_ptr<MxTools::MxpiObjectList> objectList =
        std::static_pointer_cast<MxTools::MxpiObjectList>(metaDataPtr);

    for (size_t ten_idx = 0; ten_idx < tensors.size(); ten_idx++) {
        std::vector<std::shared_ptr<void>> featLayerData;
        std::vector<std::shared_ptr<void>> featLayerData1;
        APP_ERROR ret = MemoryDataToHost(ten_idx, tensors, featLayerData);
        if (ret != APP_ERR_OK) {
            LogError << "MemoryDataToHost fail";
            return ret;
        }
        ret = MemoryDataToHost(ten_idx, tensors, featLayerData1);
        if (ret != APP_ERR_OK) {
            LogError << "MemoryDataToHost fail";
            return ret;
        }
        float *box_ptr = reinterpret_cast<float *>(featLayerData[0].get());
        float *score_ptr = reinterpret_cast<float *>(featLayerData1[1].get());

        float min_score = 0.1f;
        std::vector<std::vector<MxBase::ObjectInfo>> objectInfos(num_class);
        for (int i = 0; i < num_box; i++) {
            for (int j = 1; j < num_class; j++) {
                float score = *(score_ptr + i * num_class + j);
                if (score < min_score) continue;
                MxBase::ObjectInfo obj_info;
                obj_info.x0 =
                    (*(box_ptr + i * 4 + 1) *
                     postProcessorImageInfo.postImageInfoVec[0].widthOriginal);
                obj_info.y0 =
                    (*(box_ptr + i * 4 + 0) *
                     postProcessorImageInfo.postImageInfoVec[0].heightOriginal);
                obj_info.x1 =
                    (*(box_ptr + i * 4 + 3) *
                     postProcessorImageInfo.postImageInfoVec[0].widthOriginal);
                obj_info.y1 =
                    (*(box_ptr + i * 4 + 2) *
                     postProcessorImageInfo.postImageInfoVec[0].heightOriginal);
                obj_info.confidence = score;
                obj_info.classId = j;
                obj_info.className = configData_.GetClassName(j);
                objectInfos[j].emplace_back(obj_info);
            }
        }

        for (size_t i = 1; i < objectInfos.size(); i++) {
            objectInfos[i] = nms(objectInfos[i], 0.6f, 100);
        }

        for (size_t i = 0; i < objectInfos.size(); i++) {
            for (size_t j = 0; j < objectInfos[i].size(); j++) {
                MxBase::ObjectInfo &info = objectInfos[i][j];
                MxTools::MxpiObject *objectData = objectList->add_objectvec();
                objectData->set_x0(info.x0);
                objectData->set_y0(info.y0);
                objectData->set_x1(info.x1);
                objectData->set_y1(info.y1);
                MxTools::MxpiClass *classInfo = objectData->add_classvec();
                classInfo->set_classid(info.classId);
                classInfo->set_classname(info.className);
                classInfo->set_confidence(info.confidence);
                MxTools::MxpiMetaHeader *header = objectData->add_headervec();
                header->set_memberid(headerVec[ten_idx].memberid());
                header->set_datasource(headerVec[ten_idx].datasource());
            }
        }
    }
    return APP_ERR_OK;
}
std::vector<MxBase::ObjectInfo> MxpiSSDMobileNetV2PostProcess::nms(
    const std::vector<MxBase::ObjectInfo> &object_infos, float thres,
    int max_boxes) {
    std::vector<int> order;
    for (size_t i = 0; i < object_infos.size(); i++) {
        order.emplace_back(i);
    }

    std::sort(order.begin(), order.end(), [&object_infos](int a, int b) {
        return object_infos[a].confidence > object_infos[b].confidence;
    });

    std::vector<MxBase::ObjectInfo> res;
    while (!order.empty()) {
        res.push_back(object_infos[order[0]]);
        if (static_cast<int>(res.size()) >= max_boxes) break;
        std::vector<int> new_order;
        const MxBase::ObjectInfo &cur = object_infos[order[0]];
        for (size_t i = 1; i < order.size(); i++) {
            float ans = iou(cur, object_infos[order[i]]);
            if (ans <= thres) {
                new_order.push_back(order[i]);
            }
        }
        order = std::move(new_order);
    }

    return res;
}

float MxpiSSDMobileNetV2PostProcess::iou(const MxBase::ObjectInfo &a,
                                         const MxBase::ObjectInfo &b) {
    float xx1 = std::max(a.x0, b.x0);
    float xx2 = std::min(a.x1, b.x1);
    float yy1 = std::max(a.y0, b.y0);
    float yy2 = std::min(a.y1, b.y1);

    float w = std::max(xx2 - xx1 + 1, 0.f);
    float h = std::max(yy2 - yy1 + 1, 0.f);
    float inter = w * h;

    float area1 = (b.y1 - b.y0 + 1) * (b.x1 - b.x0 + 1);
    float area2 = (a.y1 - a.y0 + 1) * (a.x1 - a.x0 + 1);

    float ovr = inter / (area1 + area2 - inter);

    return ovr;
}
