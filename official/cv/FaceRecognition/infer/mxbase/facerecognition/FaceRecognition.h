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

#ifndef MXBASE_RESNET50CLASSIFYOPENCV_H
#define MXBASE_RESNET50CLASSIFYOPENCV_H
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "ClassPostProcessors/Resnet50PostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"


struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    uint32_t topk;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};

class FaceRecognition {
 public:
     APP_ERROR Init(const InitParam &initParam);
     APP_ERROR DeInit();
     bool ReadImage(const std::string &imgPath, cv::Mat &imageMat);
     APP_ERROR ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
     APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
     APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
     APP_ERROR PostProcess(std::vector<MxBase::TensorBase> &inputs,
     std::vector<std::vector<float>> &out);
     APP_ERROR Process(const std::string &imgPath, std::vector<std::vector<float>> &out);
     APP_ERROR main(const std::string &zj_list_path, const std::string &jk_list_path, const std::string &dis_list_path);
     float uint6_cov_float(uint16_t value);
     // get infer time
     double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}

     // lqy-function
     std::vector<int> cal_topk(int idx, std::string str_zj,
     std::map<std::string, std::vector<std::string> >& zj2jk_pairs,
     std::map<std::string, std::vector<float> >& test_embedding_tot,
     std::vector<std::vector<float> >& dis_embedding_tot_np,
     const std::map<int, std::vector<std::vector<float> > >& jk_all_embedding,
     std::vector<std::string>& dis_label);
     void l2normalize(std::vector<std::vector<float>>& out);
     int line_of_txt(std::string s);
     void deal_txt_img(std::string s, int& count, int batch_img, int out_vector_len,
     std::vector<std::vector<float>>& test_embedding_tot_np,
     std::map<std::string, std::vector<float> >& test_embedding_tot);
     void deal_txt_img_dis(std::string s, int& count, int batch_img, int out_vector_len,
     std::vector<std::vector<float>>& test_embedding_tot_np,
     std::map<std::string, std::vector<float> >& test_embedding_tot,
     std::vector<std::string>& dis_label);
     std::string get_lable_num(std::string s);
     void txt_to_pair(std::map<std::string, std::vector<std::string> >& zj2jk_pairs,
     int img_tot_zj, int img_tot_jk, std::vector<int>& ID_nums,
     const std::string &zj_list_path, const std::string &jk_list_path);
     void get_jk_all_embedding(std::vector<int>& ID_nums, std::map<std::string,
     std::vector<float> >& test_embedding_tot,
     std::map<int, std::vector<std::vector<float> > >& jk_all_embedding,
     const std::string &jk_list_path);

 private:
     APP_ERROR SaveResult(const std::string &imgPath,
     const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos, int index);

 private:
     std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
     std::shared_ptr<MxBase::Resnet50PostProcess> post_;
     MxBase::ModelDesc modelDesc_;
     uint32_t deviceId_ = 0;
     // infer time
     double inferCostTimeMilliSec = 0.0;
};


#endif

