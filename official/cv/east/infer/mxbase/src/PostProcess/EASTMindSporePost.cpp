/*
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

#include "EASTMindSporePost.h"
#include <algorithm>
#include <string>
#include <memory>
#include <utility>
#include "MxBase/Log/Log.h"
#include "lanmsUtils.h"

namespace {
    const float NMS_COEFFICIENT = 10000.0;
    const float EPS = 0.5;
    const int LIMIT_SIZE = 4;
    const int SCORE_INDEX = 0;
    const int GEO_INDEX = 1;
    const int HEIGHT_INDEX = 2;
    const int WIDTH_INDEX = 3;
}

namespace MxBase {
    APP_ERROR EASTPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
        LogInfo << "Start to Init EASTPostProcess.";
        APP_ERROR ret = TextObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogInfo << GetError(ret) << "Fail to superInit in TextObjectPostProcessBase.";
            return ret;
        }

        configData_.GetFileValue<float>("NMS_THRESH", nmsThresh_);
        configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
        configData_.GetFileValue<uint32_t>("OUT_SIZE", outSize_);

        LogInfo << "End to Init EASTPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR EASTPostProcess::DeInit() {
        LogInfo << "Begin to deinitialize EASTMindsporePost.";
        LogInfo << "End to deinitialize EASTMindsporePost.";
        return APP_ERR_OK;
    }

    bool EASTPostProcess::IsValidTensors(const std::vector<TensorBase> &tensors) const {
        if (tensors.size() != outSize_) {
            LogError << "number of tensors (" << tensors.size() << ") " << "is unequal to type("
                     << outSize_ << ")";
            return false;
        }
        return true;
    }

    void EASTPostProcess::MatrixMultiplication(const std::vector<std::vector<float>> &rotateMat,
                                               const std::vector<std::vector<float>> &coordinates,
                                               std::vector<std::vector<float>> *ret, int x, int y) {
        for (uint32_t j = 0; j < (*ret).size(); j++) {
            for (uint32_t k = 0; k < (*ret)[j].size(); k++) {
                for (uint32_t m = 0; m < (*ret).size(); m++) {
                    (*ret)[j][k] += rotateMat[j][m] * coordinates[m][k];
                }
                if (j == 0) (*ret)[j][k] += x;
                if (j == 1) (*ret)[j][k] += y;
            }
        }
    }

    void EASTPostProcess::RestorePolys(const std::vector<TensorBase>& tensors,
                                       const std::vector<std::vector<uint32_t>> &validPos,
                                       const std::vector<std::vector<uint32_t>> &xyText,
                                       const std::vector<std::vector<float>> &validGeo,
                                       std::vector<std::vector<float>> *polys) {
        auto bboxPtr = reinterpret_cast<float *>(tensors[SCORE_INDEX].GetBuffer());
        auto scoreShape = tensors[SCORE_INDEX].GetShape();

        for (uint32_t i = 0; i < validGeo[0].size(); i++) {
            int x = validPos[i][0];
            int y = validPos[i][1];
            float yMin = y - validGeo[0][i];
            float yMax = y + validGeo[1][i];
            float xMin = x - validGeo[2][i];
            float xMax = x + validGeo[3][i];

            float theta = -validGeo[4][i];
            std::vector<std::vector<float>> rotateMat = {{cos(theta), -sin(theta)},
                                                         {sin(theta), cos(theta)}};
            std::vector<float> tempX = {xMin - x, xMax - x, xMax - x, xMin - x};
            std::vector<float> tempY = {yMin - y, yMin - y, yMax - y, yMax - y};
            std::vector<std::vector<float>> coordinates = {tempX, tempY};
            std::vector<std::vector<float>> ret(rotateMat.size(), std::vector<float>(coordinates[0].size()));
            MatrixMultiplication(rotateMat, coordinates, &ret, x, y);

            int cnt = 0;
            for (uint32_t j = 0; j < ret[0].size(); j++) {
                if (ret[0][j] < 0 || ret[0][j] >= (scoreShape[WIDTH_INDEX] * LIMIT_SIZE) || \
                    ret[1][j] < 0 || ret[1][j] >= (scoreShape[HEIGHT_INDEX] * LIMIT_SIZE)) cnt += 1;
            }

            if (cnt <= 1) {
                (*polys).push_back(std::vector<float>{ret[0][0] * NMS_COEFFICIENT, ret[1][0] * NMS_COEFFICIENT,
                                                      ret[0][1] * NMS_COEFFICIENT, ret[1][1] * NMS_COEFFICIENT,
                                                      ret[0][2] * NMS_COEFFICIENT, ret[1][2] * NMS_COEFFICIENT,
                                                      ret[0][3] * NMS_COEFFICIENT, ret[1][3] * NMS_COEFFICIENT,
                                                      bboxPtr[xyText[i][0] * scoreShape[WIDTH_INDEX] + xyText[i][1]]});
            }
        }
    }

    void EASTPostProcess::GetXYText(const std::vector<TensorBase>& tensors,
                                    std::vector<std::vector<uint32_t>> *xyText) {
        auto bboxPtr = reinterpret_cast<float *>(tensors[SCORE_INDEX].GetBuffer());
        auto scoreShape = tensors[SCORE_INDEX].GetShape();

        for (uint32_t i = 0; i < scoreShape[HEIGHT_INDEX]; i++) {
            for (uint32_t j = 0; j < scoreShape[WIDTH_INDEX]; j++) {
                if (bboxPtr[i * scoreShape[WIDTH_INDEX] + j] > scoreThresh_) {
                    (*xyText).push_back({i, j});
                }
            }
        }
    }

    void EASTPostProcess::GetValidGeo(const std::vector<TensorBase> &tensors,
                                      const std::vector<std::vector<uint32_t>> &xyText,
                                      std::vector<std::vector<float>> *validGeo) {
        auto dataPtr = reinterpret_cast<float *>(tensors[1].GetBuffer());
        auto geoShape = tensors[GEO_INDEX].GetShape();
        (*validGeo).resize(geoShape[GEO_INDEX]);

        uint32_t lens = geoShape[HEIGHT_INDEX] * geoShape[WIDTH_INDEX];
        for (auto item : xyText) {
            for (uint32_t i = 0; i < (*validGeo).size(); i++) {
                (*validGeo)[i].push_back(dataPtr[i * lens + item[0] * geoShape[WIDTH_INDEX] + item[1]]);
            }
        }
    }

    void EASTPostProcess::GetTextObjectInfo(const std::vector<std::vector<float>> &polys,
                                            const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                                            std::vector<TextObjectInfo> *textsInfos) {
        auto boxes = lanms::MergeQuadrangleN9(polys, nmsThresh_);
        std::vector<std::vector<float>> ans;
        for (size_t i = 0; i < boxes.size(); i++) {
            auto &p = boxes[i];
            auto &poly = p.poly;
            ans.push_back(std::vector<float>{
                    static_cast<float>(poly[0].X / NMS_COEFFICIENT), static_cast<float>(poly[0].Y / NMS_COEFFICIENT),
                    static_cast<float>(poly[1].X / NMS_COEFFICIENT), static_cast<float>(poly[1].Y / NMS_COEFFICIENT),
                    static_cast<float>(poly[2].X / NMS_COEFFICIENT), static_cast<float>(poly[2].Y / NMS_COEFFICIENT),
                    static_cast<float>(poly[3].X / NMS_COEFFICIENT), static_cast<float>(poly[3].Y / NMS_COEFFICIENT),
                    static_cast<float>(p.score),
            });
        }

        // resize boxes
        auto resizedImageInfo = resizedImageInfos[0];
        float heightRatio = resizedImageInfo.heightResize * 1.0 / resizedImageInfo.heightOriginal;
        float widthRatio = resizedImageInfo.widthResize * 1.0 / resizedImageInfo.widthOriginal;
        for (auto& box : ans) {
            for (uint32_t j = 0; j < box.size() - 1; j++) {
                if (j % 2 == 1) {
                    box[j] /= heightRatio;
                } else {
                    box[j] /= widthRatio;
                }
            }
        }

        TextObjectInfo info;
        for (auto box : ans) {
            info.x0 = box[0] + EPS; info.y0 = box[1] + EPS;
            info.x1 = box[2] + EPS; info.y1 = box[3] + EPS;
            info.x2 = box[4] + EPS; info.y2 = box[5] + EPS;
            info.x3 = box[6] + EPS; info.y3 = box[7] + EPS;
            info.confidence = box[8];
            (*textsInfos).push_back(info);
        }
    }

    void EASTPostProcess::GetValidDetBoxes(const std::vector<TensorBase> &tensors,
                                           const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                                           std::vector<TextObjectInfo> *textsInfos) {
        LogInfo << "Begin to GetValidDetBoxes.";

        std::vector<std::vector<uint32_t>> xyText;
        GetXYText(tensors, &xyText);

        std::vector<std::vector<uint32_t>> validPos(xyText.size());
        std::transform(xyText.begin(), xyText.end(), validPos.begin(),
                       [&](const std::vector<uint32_t> &item) { return item; });
        for (auto &item : validPos) {
            std::swap(item[0], item[1]);
            item[0] = item[0] * LIMIT_SIZE;
            item[1] = item[1] * LIMIT_SIZE;
        }

        std::vector<std::vector<float>> validGeo;
        GetValidGeo(tensors, xyText, &validGeo);

        std::vector<std::vector<float>> polys;
        RestorePolys(tensors, validPos, xyText, validGeo, &polys);

        GetTextObjectInfo(polys, resizedImageInfos, textsInfos);
    }

    APP_ERROR EASTPostProcess::Process(const std::vector<MxBase::TensorBase>& tensors,
                                       std::vector<std::vector<MxBase::TextObjectInfo>> &textObjInfos,
                                       const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                                       const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
        LogDebug << "Start to Process EASTPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << "CheckAndMoveTensors failed. ret=" << ret;
            return ret;
        }

        ObjectDetectionOutput(tensors, &textObjInfos, resizedImageInfos);

        LogDebug << "End to Process EASTPostProcess.";
        return APP_ERR_OK;
    }

    void EASTPostProcess::ObjectDetectionOutput(const std::vector<MxBase::TensorBase>& tensors,
                                                std::vector<std::vector<MxBase::TextObjectInfo>> *textObjInfos,
                                                const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos) {
        LogInfo << "EASTMindsporePost start to write results.";
        auto shape = tensors[0].GetShape();
        uint32_t batchSize = shape[0];
        for (uint32_t i = 0; i < batchSize; i++) {
            std::vector<TextObjectInfo> textsInfo;
            GetValidDetBoxes(tensors, resizedImageInfos, &textsInfo);
            (*textObjInfos).push_back(textsInfo);
        }

        LogInfo << "EASTMindsporePost write results succeeded.";
    }

#ifndef ENABLE_POST_PROCESS_INSTANCE
    extern "C" {
    std::shared_ptr<MxBase::EASTPostProcess> GetTextObjectInstance() {
        LogInfo << "Begin to get EASTPostProcess instance.";
        auto instance = std::make_shared<EASTPostProcess>();
        LogInfo << "End to get EASTPostProcess instance.";
        return instance;
    }
    }
#endif

}  // namespace MxBase
