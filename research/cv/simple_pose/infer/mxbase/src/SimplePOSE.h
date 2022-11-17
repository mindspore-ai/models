/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef SimplePOSEPOST_SimplePOSE_H
#define SimplePOSEPOST_SimplePOSE_H


#include<vector>
#include<string>
#include<memory>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "PostProcess/SimplePOSEMindsporePost.h"
#include "MxBase/DeviceManager/DeviceManager.h"

struct InitParam {
    uint32_t deviceId;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};

struct ImageShape {
    uint32_t width;
    uint32_t height;
};

class SimplePOSE {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string& imgPath, cv::Mat *imageMat, ImageShape *imgShape);
    APP_ERROR Resize_Affine(const cv::Mat& srcImage, cv::Mat *dstImage, ImageShape *imgShape,
        const float center[], const float scale[]);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);

    APP_ERROR CVMatToTensorBase(const cv::Mat& imageMat, MxBase::TensorBase *tensorBase);
    APP_ERROR CVMatToTensorBaseFlip(const cv::Mat& imageMat, MxBase::TensorBase* tensorBase);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
        const std::vector<MxBase::TensorBase>& inputs1,
        std::vector<std::vector<float> >* node_score_list, const float center[], const float scale[]);
    APP_ERROR Process(const std::string& BBOX_FILE, const std::string &imgPath, const std::string &resultPath);

 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::SimplePOSEMindsporePost> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
};

#endif  // SimplePOSEPOST_SimplePOSE_H
