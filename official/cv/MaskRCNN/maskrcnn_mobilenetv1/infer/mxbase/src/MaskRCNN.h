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

#ifndef MASKRCNNPOST_MASKRCNN_H
#define MASKRCNNPOST_MASKRCNN_H

#include <vector>
#include <map>
#include <string>
#include <memory>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "PostProcess/MaskRcnnMindsporePost.h"
#include "MxBase/DeviceManager/DeviceManager.h"


struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    float iou_thresh;
    float score_thresh;

    bool checkTensor;
    std::string modelPath;
};

struct ImageShape {
    uint32_t width;
    uint32_t height;
};

class MaskRCNN {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, MxBase::DvppDataInfo &output, ImageShape &imgShape);
    APP_ERROR Resize(const MxBase::DvppDataInfo &input, MxBase::TensorBase &outputTensor);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                          std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                          const std::map<std::string, std::shared_ptr<void>> &configParamMap);
    APP_ERROR Process(const std::string &imgPath, const std::string &resultPath);

 private:
    APP_ERROR GetImageMeta(const ImageShape &imageShape, MxBase::TensorBase &imgMetas) const;
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::MaskRcnnMindsporePost> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
};

#endif  // MASKRCNNPOST_MASKRCNN_H
