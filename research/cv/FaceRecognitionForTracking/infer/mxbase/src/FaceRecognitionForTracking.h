/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef FACE_RECOGNITION_FOR_TRACKING_H
#define FACE_RECOGNITION_FOR_TRACKING_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class FaceRecognitionForTracking {
 public:
    /**
     * @api
     * @brief Initialize configure parameter.
     * @param initParam
     * @return APP_ERROR
     */
    APP_ERROR Init(const InitParam &initParam);
    /**
     * @api
     * @brief DeInitialize configure parameter.
     * @return APP_ERROR
     */
    APP_ERROR DeInit();
    /**
     * @api
     * @brief reads a jpg image and convert the BGR format to RGB format.
     * @param imgPath: the image path
     * @param imgMat: the image matrix stored as cv::Mat
     * @return APP_ERROR
     */
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imgMat);
    /**
     * @api
     * @brief resize the origin image to size(96,64); 
     *        do transpose(2,0,1) to convert format HWC to CHW;
     *        do normalized to [-1, 1];
     *        convert cv::Mat to float*(1-d float array)
     * @param srcMat: source image matrix(cv::Mat) from ReadImage
     * @param dstMat: an 1-d float array pointer to store the image matrix info
     * @return APP_ERROR
     */
    APP_ERROR Resize(const cv::Mat &srcMat, float *dstMat);
    /**
     * @api
     * @brief convert the 1-d float array to TensorBase
     * @param imgMat: an 1-d float array pointer stored the image matrix info
     * @param tensorBase: the input of infer model
     * @return APP_ERROR
     */
    APP_ERROR CvMatToTensorBase(float* imgMat, MxBase::TensorBase &tensorBase);
    /**
     * @api
     * @brief perform inference process
     * @param inputs: the input vector Tensorbase vector
     * @param outputs: the output vector of Tensorbase vector
     * @return APP_ERROR
     */
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    /**
     * @api
     * @brief calculate the result
     * @param inputs: the input vector Tensorbase vector
     * @param names: the images' name
     * @return APP_ERROR
     */
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs, std::vector<std::string> &names);
    /**
     * @api
     * @brief calculate the accuracy
     * @param imgPath: test dataset directory
     * @return APP_ERROR
     */
    APP_ERROR Process(const std::string &imgPath);
    /**
     * @api
     * @brief get the infer time
     * @return double
     */
    double GetInferCostTimeMs() const {
        return inferCostTimeMs;
    }

 private:
    std::vector<std::string> GetFileList(const std::string &dirPath);
    void InclassLikehood(cv::Mat &featureMat, std::vector<std::string> &names, std::vector<float> &inclassLikehood);
    void BtclassLikehood(cv::Mat &featureMat, std::vector<std::string> &names, std::vector<float> &btclassLikehood);
    void TarAtFar(std::vector<float> &inclassLikehood, std::vector<float> &btclassLikehood, \
        std::vector<std::vector<float>> &tarFars);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    double inferCostTimeMs = 0.0;
};

#endif
