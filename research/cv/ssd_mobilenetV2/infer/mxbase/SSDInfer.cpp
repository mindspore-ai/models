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

#include "infer/mxbase/SSDInfer.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>

#include "boost/filesystem.hpp"
#include "infer/mxbase/MxImage.h"
#include "infer/mxbase/MxUtil.h"

namespace sdk_infer {
namespace mxbase_infer {

SSDInfer::SSDInfer(uint32_t device_id, std::string classes_path,
                   std::string classes_idx_path, std::string save_dir,
                   bool is_debug, int width, int height,
                   std::string config_cfg_path, std::string label_path)
    : MxBaseInfer(device_id),
      classes_path_(classes_path),
      classes_idx_path_(classes_idx_path),
      save_dir_(save_dir),
      is_debug_(is_debug),
      model_width_(width),
      model_height_(height),
      config_cfg_path_(config_cfg_path),
      label_path_(label_path) {}

bool SSDInfer::AfterInit() {
    sdk_infer::mxbase_infer::MxUtil util;
    util.LogModelDesc(m_model_processor, m_model_desc);

    {
        if (!boost::filesystem::is_regular_file(classes_path_)) {
            LogInfo << "read from : " << classes_path_ << " fail";
            return false;
        }
        std::fstream classes_stream(classes_path_);
        classes_ = web::json::value::parse(classes_stream);
        LogInfo << "Read classes from : " << classes_path_ << classes_;
    }
    {
        if (!boost::filesystem::is_regular_file(classes_idx_path_)) {
            LogInfo << "read from : " << classes_idx_path_ << " fail";
            return false;
        }
        std::fstream classes_idx_stream(classes_idx_path_);
        web::json::value tmp = web::json::value::parse(classes_idx_stream);
        LogInfo << "read from" << classes_idx_path_ << " " << tmp;
        for (size_t i = 0; i < tmp.size(); i++) {
            classes_idx_.push_back(tmp[i].as_integer());
        }
    }

    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigPath"] =
        std::make_shared<std::string>(config_cfg_path_);
    config["labelPath"] = std::make_shared<std::string>(label_path_);
    m_post_process.Init(config);

    return true;
}

void SSDInfer::SaveResult() {
    {
        web::json::value output = web::json::value::array(preditions);
        std::fstream out(save_dir_ + "predictions.json",
                         std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            LogError << "write preditions.json error";
            return;
        }
        out << output.serialize();
    }
    {
        web::json::value image_id_output = web::json::value::array(images_ids);
        std::fstream out1(save_dir_ + "imageid.json",
                          std::ios::out | std::ios::trunc);
        if (!out1.is_open()) {
            LogError << "write imageid.json error";
            return;
        }
        out1 << image_id_output.serialize();
    }
}

bool SSDInfer::PrepareImages(const cv::Mat &origin_image, cv::Mat *out_image,
                             uint32_t model_width, uint32_t model_height) {
    resize_info.heightOriginal = origin_image.rows;
    resize_info.widthOriginal = origin_image.cols;
    resize_info.heightResize = model_height;
    resize_info.widthResize = model_width;
    cv::Size dsize = cv::Size(model_width, model_height);
    *out_image = cv::Mat(dsize, origin_image.type());
    cv::resize(origin_image, *out_image, dsize);
    cv::cvtColor(*out_image, *out_image, static_cast<int>(cv::COLOR_BGR2RGB));
    return true;
}

void SSDInfer::SaveObjectInfo(
    std::vector<std::vector<MxBase::ObjectInfo>> &object_infos) {
    cv::Mat images;
    images = cv::imread(m_file_path);
    if (images.empty()) {
        LogError << "read images fail " << m_file_path << std::endl;
        return;
    }
    for (size_t j = 1; j < object_infos.size(); j++)
        for (size_t i = 0; i < object_infos[j].size(); i++) {
            MxBase::ObjectInfo object_info = object_infos[j][i];
            int x0 = object_info.x0;
            int y0 = object_info.y0;
            int x1 = object_info.x1;
            int y1 = object_info.y1;
            if (object_info.confidence < 0.4) continue;

            cv::Scalar color(255, 0, 0);
            cv::rectangle(images, cv::Point2i(x0, y0), cv::Point2i(x1, y1),
                          color, 1);
            cv::putText(images, object_info.className,
                        cv::Point(x0 + 5, std::max<int>(y0 - 5, 0)), 2, 0.5,
                        cv::Scalar(0, 0, 255), 1);
        }
    cv::imwrite(save_dir_ +
                    boost::filesystem::path(m_file_path).stem().c_str() +
                    ".jpg",
                images);
}

void SSDInfer::PostProcess(std::vector<MxBase::TensorBase> &output_tensor) {
    std::vector<std::vector<MxBase::ObjectInfo>> object_infos;
    m_post_process.Process(output_tensor, object_infos, {resize_info});

    for (auto &objs : object_infos)
        for (auto &obj : objs) {
            web::json::value tmp;
            tmp["image_id"] = m_image_id;
            web::json::value bbox = web::json::value::array(
                {obj.x0, obj.y0, (obj.x1 - obj.x0), (obj.y1 - obj.y0)});
            tmp["bbox"] = bbox;
            tmp["score"] = obj.confidence;
            tmp["category_id"] = classes_idx_[static_cast<int>(obj.classId)];
            preditions.push_back(tmp);
        }

    if (is_debug_) SaveObjectInfo(object_infos);
}

bool SSDInfer::LoadImageToModel(const std::string &file,
                                        MxBase::TensorBase *tensor,
                                        size_t image_id) {
    m_file_path = file;
    m_image_id = image_id;
    images_ids.push_back(image_id);

    cv::Mat image = cv::imread(file);
    if (image.empty()) {
        LogError << "read image from " << file << "fail" << std::endl;
        return false;
    }

    uint32_t model_width;
    uint32_t model_height;
    GetWidthAndHeightFromModel(&model_width, &model_height, 0);

    cv::Mat out_image;
    bool is_ok = PrepareImages(image, &out_image, model_width, model_height);

    if (!is_ok) {
        LogError << "PrepareImages fail.";
        return false;
    }

    cv::Mat device_mat;
    sdk_infer::mxbase_infer::MxImage image_tool;
    image_tool.ConvertToDeviceFormat(
        out_image, (aclDataType)m_model_processor.GetInputDataType()[0],
        (aclFormat)m_model_processor.GetInputFormat()[0], &device_mat);

    cv::Scalar means = {0.485, 0.456, 0.406};
    cv::Scalar stds = {0.229, 0.224, 0.225};
    assert(device_mat.dims == 4 && device_mat.size[0] == 1 &&
           device_mat.type() == CV_32F);
    size_t plane = device_mat.step1(1);
    // first channel
    float *ptr = reinterpret_cast<float *>(device_mat.ptr(0));
    for (size_t i = 0; i < plane; i++) {
        *ptr = (*ptr - means[0]) / stds[0];
        ptr++;
    }
    // second channel
    for (size_t i = 0; i < plane; i++) {
        *ptr = (*ptr - means[1]) / stds[1];
        ptr++;
    }
    // third channel
    for (size_t i = 0; i < plane; i++) {
        *ptr = (*ptr - means[2]) / stds[2];
        ptr++;
    }

    if (device_mat.total() * device_mat.elemSize() !=
        m_model_desc.inputTensors[0].tensorSize) {
        LogError << "device_mat'size is diff to m_memory_inputs'size ";
        return false;
    }

    MxBase::MemoryData data(m_model_desc.inputTensors[0].tensorSize,
                            MxBase::MemoryData::MEMORY_DEVICE, m_deviceId);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMalloc(data);

    if (ret != APP_ERR_OK) {
        LogError << "LoadImageAsInput MxbsMalloc fail";
        return false;
    }

    MxBase::MemoryData tmp(device_mat.data, data.size);
    MxBase::MemoryHelper::MxbsMemcpy(data, tmp, tmp.size);

    auto shape = m_model_processor.GetInputShape()[0];
    *tensor = MxBase::TensorBase(
        data, false, std::vector<uint32_t>{shape.begin(), shape.end()},
        m_model_processor.GetInputDataType()[0]);
    return true;
}

}  // namespace mxbase_infer
}  // namespace sdk_infer
