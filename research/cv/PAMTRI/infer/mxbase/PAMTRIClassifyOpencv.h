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

#ifndef MXBASE_PAMTRICLASSIFYOPENCV_H
#define MXBASE_PAMTRICLASSIFYOPENCV_H

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "opencv4/opencv2/opencv.hpp"
struct InitParam {
    uint32_t deviceId;
    std::string PoseEstNetPath;
    std::string MultiTaskNetPath;
    std::string resultPath;
    int batchSize;
    bool segmentAware;
    bool heatmapAware;
};
class PAMTRIClassifyOpencv {
 public:
    explicit PAMTRIClassifyOpencv(const InitParam &InitParam);
    APP_ERROR init_(const InitParam &initParam);
    APP_ERROR deinit_();
    void read_image(const std::string &img_path, cv::Mat *image_mat);
    cv::Mat get_affinetransform(const cv::Point2f &center,
                                const cv::Point2f &scale, const float rot,
                                const cv::Size &out_size, const cv::Point2f &shift,
                                bool inv);
    APP_ERROR flip_(const cv::Mat &pn_input, MxBase::TensorBase *orig_output);
    APP_ERROR get_posenet_input(const std::vector<std::string> &img_path,
                                    cv::Mat *pn_input,
                                    std::vector<cv::Point2f> *center,
                                    std::vector<cv::Point2f> *scale,
                                    std::vector<cv::Mat> *mt_rgb_inputs);
    APP_ERROR get_posenet_pred(const MxBase::TensorBase &pn_output,
                                std::vector<cv::Point2f> *pred,
                                std::vector<float> *maxvals,
                                const std::vector<cv::Point2f> &center,
                                const std::vector<cv::Point2f> &scale);
    APP_ERROR get_multitask_input(const std::vector<cv::Mat> &rgb_mat,
                                    const MxBase::TensorBase &pn_output,
                                    const std::vector<cv::Point2f> &kpt_pred,
                                    const std::vector<float> &maxvals,
                                    cv::Mat *multiTask_img_input, cv::Mat *vkpts);
    void preprocess_image(const cv::Mat &src_bgr_mat, cv::Mat *dst_rgb_mat,
                                cv::Mat *mt_rgb_input);
    APP_ERROR mat_to_tensorbase(const cv::Mat &src_mat,
                                MxBase::TensorBase *dst_tensorBase,
                                std::vector<int> src_shape);
    APP_ERROR posenet_infer(const std::vector<MxBase::TensorBase> &inputs,
                                    std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR multitask_infer(const std::vector<MxBase::TensorBase> &inputs,
                                    std::vector<MxBase::TensorBase> *outputs);
    APP_ERROR process_(const std::vector<std::string> &imgPath);
    APP_ERROR save_result(const std::vector<std::string> &imgPath,
                        const std::vector<MxBase::TensorBase> &mt_outputs);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_poseest_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_multitask_;
    MxBase::ModelDesc poseestnet_desc_;
    MxBase::ModelDesc multitasknet_desc_;
    std::vector<int> mt_img_input_shape_;
    std::vector<int> mt_kpt_input_shape_;
    std::vector<int> pn_input_shape_;
    bool segment_aware_;
    bool heatmap_aware_;
    int batch_size_;
    std::ofstream result_file_;
    uint32_t deviceId_ = 0;
};

#endif
