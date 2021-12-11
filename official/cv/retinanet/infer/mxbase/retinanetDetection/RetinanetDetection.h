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

#ifndef MXBASE_RETINANETDETECTION_H
#define MXBASE_RETINANETDETECTION_H

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_infer_cost;

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    std::string modelPath;
    uint32_t resizeHeight;
    uint32_t resizeWidth;
    std::vector<std::string> label;
    uint32_t width;
    uint32_t height;
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<float>> scores;
    uint32_t maxBoxes;
    uint32_t nmsThershold;
    uint32_t minScore;
    uint32_t numRetinanetBoxes;
    uint32_t classNum;
    uint32_t boxDim;
};

class RetinanetDetection {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR process(const std::string &imgName, InitParam &initParam);

 protected:
    APP_ERROR read_image(const std::string &imgPath, cv::Mat &imageMat);
    APP_ERROR resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat, const InitParam &initParam);
    APP_ERROR cvmat_to_tensorbase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
    APP_ERROR get_tensor_output(size_t index, MxBase::TensorBase output, InitParam &initParam);
    APP_ERROR get_anm_result(InitParam &initParam);
    APP_ERROR post_process(std::vector<MxBase::TensorBase> outputs, InitParam &initParam);
    APP_ERROR write_result(std::vector<std::vector<std::vector<float>>> final_boxes,
                           std::vector<std::vector<float>> final_score,
                           std::vector<std::vector<std::string>> final_label);
    APP_ERROR apply_nms(std::vector<std::vector<float>> &class_boxes,
                        std::vector<float> &class_box_scores, std::vector<int> &keep, const InitParam &initParam);
    APP_ERROR get_column_data(std::vector<float> &get_vector,
                              std::vector<std::vector<float>> &input_vector, int index);
    APP_ERROR maxiMum(float x, std::vector<float> &other_x,
                      std::vector<int> &order, std::vector<float> &get_x);
    APP_ERROR miniMum(float x, std::vector<float> &other_x,
                      std::vector<int> &order, std::vector<float> &get_x);
    APP_ERROR get_order_data(const std::vector<float> &areas, const std::vector<float> &inter,
                             std::vector<int> &order, const InitParam &initParam);

 private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_ = {};
    uint32_t deviceId_ = 0;
};
#endif
