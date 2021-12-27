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

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "MxBase/Log/Log.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace sdk_infer {
namespace mxbase_infer {

class SSDPostProcessor : public MxBase::ObjectPostProcessBase {
 public:
    SSDPostProcessor(const SSDPostProcessor &r) {
        LogInfo << "copy & SSDPostProcessor" << this << " " << &r;
    }
    SSDPostProcessor() { LogInfo << "SSDPostProcessor()" << this << " "; }

    APP_ERROR Process(
        const std::vector<MxBase::TensorBase> &tensors,
        std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
        const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
        const std::map<std::string, std::shared_ptr<void>> & = {}) override {
        auto input = tensors;
        APP_ERROR ret = ObjectPostProcessBase::CheckAndMoveTensors(input);
        if (ret != APP_ERR_OK) {
            LogInfo << "PostProcess error ret=" << ret;
            return ret;
        }
        auto &box_tensor = tensors[0];
        auto &scores_tensor = tensors[1];

        int num_class = scores_tensor.GetShape()[2];
        int num_box = box_tensor.GetShape()[1];
        float *box_ptr = reinterpret_cast<float *>(box_tensor.GetBuffer());
        float *score_ptr = reinterpret_cast<float *>(scores_tensor.GetBuffer());

        float min_score = 0.1f;
        objectInfos.resize(num_class);
        float widthOriginal = 1.f;
        float heightOriginal = 1.f;
        float has_origin_info = 0.f;

        if (resizedImageInfos.size() > 0) {
            widthOriginal = resizedImageInfos[0].widthOriginal;
            heightOriginal = resizedImageInfos[0].heightOriginal;
            has_origin_info = 1.f;
        }

        for (int i = 0; i < num_box; i++) {
            for (int j = 1; j < num_class; j++) {
                float score = *(score_ptr + i * num_class + j);
                if (score < min_score) continue;
                MxBase::ObjectInfo obj_info;
                obj_info.x0 = (*(box_ptr + i * 4 + 1) * widthOriginal);
                obj_info.y0 = (*(box_ptr + i * 4 + 0) * heightOriginal);
                obj_info.x1 = (*(box_ptr + i * 4 + 3) * widthOriginal);
                obj_info.y1 = (*(box_ptr + i * 4 + 2) * heightOriginal);
                obj_info.confidence = score;
                obj_info.classId = j;
                obj_info.className = configData_.GetClassName(j);
                objectInfos[j].emplace_back(obj_info);
            }
        }

        for (size_t i = 1; i < objectInfos.size(); i++) {
            objectInfos[i] = nms(objectInfos[i], 0.6f, 100, has_origin_info);
        }

        return APP_ERR_OK;
    }

    std::vector<MxBase::ObjectInfo> nms(
        const std::vector<MxBase::ObjectInfo> &object_infos, float thres,
        int max_boxes, float has_origin_info) {
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
                float ans = iou(cur, object_infos[order[i]], has_origin_info);
                if (ans <= thres) {
                    new_order.push_back(order[i]);
                }
            }
            order = std::move(new_order);
        }

        return res;
    }

    float iou(const MxBase::ObjectInfo &a, const MxBase::ObjectInfo &b,
              float has_origin_info) {
        float xx1 = std::max(a.x0, b.x0);
        float xx2 = std::min(a.x1, b.x1);
        float yy1 = std::max(a.y0, b.y0);
        float yy2 = std::min(a.y1, b.y1);

        float w = std::max(xx2 - xx1 + has_origin_info, 0.f);
        float h = std::max(yy2 - yy1 + has_origin_info, 0.f);
        float inter = w * h;

        float area1 =
            (b.y1 - b.y0 + has_origin_info) * (b.x1 - b.x0 + has_origin_info);
        float area2 =
            (a.y1 - a.y0 + has_origin_info) * (a.x1 - a.x0 + has_origin_info);

        float ovr = inter / (area1 + area2 - inter);

        return ovr;
    }
};  // namespace mxbase_infer

}  // namespace mxbase_infer
}  // namespace sdk_infer
