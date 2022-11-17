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

#ifndef EAST_POST_PROCESS_H
#define EAST_POST_PROCESS_H
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/PostProcessBases/TextObjectPostProcessBase.h"

namespace MxBase {
class EASTPostProcess : public TextObjectPostProcessBase {
 public:
    EASTPostProcess() = default;

    ~EASTPostProcess() = default;

    EASTPostProcess(const EASTPostProcess &other) = default;

    EASTPostProcess &operator=(const EASTPostProcess &other);

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

    APP_ERROR DeInit() override;

    APP_ERROR Process(const std::vector<MxBase::TensorBase>& tensors,
                      std::vector<std::vector<MxBase::TextObjectInfo>>& textObjInfos,
                      const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                      const std::map<std::string, std::shared_ptr<void>> &configParamMap = {}) override;

 protected:
    bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

 private:
    void MatrixMultiplication(const std::vector<std::vector<float>> &rotateMat,
                              const std::vector<std::vector<float>> &coordidates,
                              std::vector<std::vector<float>> *ret, int x, int y);
    void RestorePolys(const std::vector<TensorBase>& tensors,
                      const std::vector<std::vector<uint32_t>> &validPos,
                      const std::vector<std::vector<uint32_t>> &xyText,
                      const std::vector<std::vector<float>> &validGeo,
                      std::vector<std::vector<float>> *polys);
    void GetXYText(const std::vector<TensorBase>& tensors,
                   std::vector<std::vector<uint32_t>> *xyText);
    void GetValidGeo(const std::vector<TensorBase>& tensors,
                     const std::vector<std::vector<uint32_t>> &xyText,
                     std::vector<std::vector<float>> *validGeo);
    void GetValidDetBoxes(const std::vector<TensorBase> &tensors,
                          const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                          std::vector<TextObjectInfo> *textsInfos);
    void GetTextObjectInfo(const std::vector<std::vector<float>> &polys,
                           const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                           std::vector<TextObjectInfo> *textsInfos);
    void ObjectDetectionOutput(const std::vector<MxBase::TensorBase>& tensors,
                               std::vector<std::vector<MxBase::TextObjectInfo>>* textObjInfos,
                               const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos);

 protected:
    float nmsThresh_;
    float scoreThresh_;
    uint32_t outSize_ = 2;
};
#ifndef ENABLE_POST_PROCESS_INSTANCE
    extern "C" {
        std::shared_ptr<MxBase::EASTPostProcess> GetTextObjectInstance();
    }
#endif
}  // namespace MxBase
#endif
