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

#include <string>
#include <vector>

#include "infer/mxbase/MxBaseInfer.h"
#include "infer/mxbase/SSDPostProcessor.h"

namespace sdk_infer {
namespace mxbase_infer {

class SSDInfer : public MxBaseInfer {
 public:
    SSDInfer(uint32_t device_id, std::string classes_path,
             std::string classes_idx_path, std::string save_dir, bool is_debug,
             int model_width, int model_height,
             std::string config_cfg_path, std::string label_path);

    bool AfterInit() override;

    bool PrepareImages(const cv::Mat &origin_image, cv::Mat *out_image,
                       uint32_t model_width, uint32_t model_height);

    void SaveResult();

    void PostProcess(std::vector<MxBase::TensorBase> &output_tensor) override;

    bool LoadImageToModel(const std::string &file, MxBase::TensorBase *tensor,
                          size_t image_id) override;

 private:
    void SaveObjectInfo(std::vector<std::vector<MxBase::ObjectInfo>> &);

    sdk_infer::mxbase_infer::SSDPostProcessor m_post_process;
    std::vector<web::json::value> preditions;
    MxBase::ResizedImageInfo resize_info;

    // 所有处理图像的id
    std::vector<web::json::value> images_ids;

    // 当前处理图像的信息
    std::string m_file_path;
    size_t m_image_id;

    // 类型相关数据
    std::string classes_path_;
    web::json::value classes_;
    std::string classes_idx_path_;
    std::vector<int> classes_idx_;

    // save dir
    std::string save_dir_;

    // is bebug
    bool is_debug_;

    int model_width_;
    int model_height_;

    std::string config_cfg_path_;
    std::string label_path_;
};

}  // namespace mxbase_infer
}  // namespace sdk_infer
