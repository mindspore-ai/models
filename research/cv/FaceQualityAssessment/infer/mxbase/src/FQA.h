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
#ifndef UNET_SEGMENTATION_H
#define UNET_SEGMENTATION_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class FQA {
 public:
    /**
     * @api
     * @brief Initialize configure parameter.
     * @param initParam
     * @return APP_ERROR
     */
    APP_ERROR Init(const InitParam& initParam);
    /**
     * @api
     * @brief DeInitialize configure parameter.
     * @return APP_ERROR
     */
    APP_ERROR DeInit();
    /**
     * @api
     * @brief reads a jpg image and convert the BGR format to RGB format.
     * @param imgPath: 
     * @param imageMat: the image matrix stored as cv::Mat
     * @return APP_ERROR
     */
    APP_ERROR ReadImage(const std::string& imgPath, cv::Mat& imageMat);
    /**
     * @api
     * @brief resize the origin image to size(96,96); 
     *        do transpose(2,0,1) to convert format HWC to CHW;
     *        convert cv::Mat to float*(1-d float array)
     * @param srcImageMat: source image matrix(cv::Mat) from ReadImage
     * @param transMat: an 1-d float array pointer to store the image matrix info
     * @return APP_ERROR
     */
    APP_ERROR ResizeImage(const cv::Mat& srcImageMat, float* transMat);
    /**
     * @api
     * @brief convert the 1-d float array to TensorBase
     * @param transMat: an 1-d float array pointer stored the image matrix info
     * @param tensorBase: the input of infer model
     * @return APP_ERROR
     */
    APP_ERROR VectorToTensorBase(float* transMat, MxBase::TensorBase& tensorBase);
    /**
     * @api
     * @brief perform inference process
     * @param inputs: the input vector Tensorbase vector
     * @param outputs: the output vector of Tensorbase vector
     * @return APP_ERROR
     */
    APP_ERROR Inference(const std::vector<MxBase::TensorBase>& inputs, std::vector<MxBase::TensorBase>& outputs);
    APP_ERROR PreProcess(const std::string& imgPath, std::vector<MxBase::TensorBase>& inputs, int& width, int& height);
    /**
     * @api
     * @brief calculate the accuracy
     * @param testPath: test dataset directory
     * @return APP_ERROR
     */
    APP_ERROR Process(const std::string& testPath);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
};

#endif
