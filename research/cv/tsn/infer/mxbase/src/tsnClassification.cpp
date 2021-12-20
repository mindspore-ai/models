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

#include <unistd.h>
#include <sys/stat.h>
#include <opencv2/imgproc/types_c.h>
#include <memory>
#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include "tsnClassification.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"


APP_ERROR TSN::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }

    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR TSN::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


APP_ERROR TSN::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i],  MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        (*outputs).push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, (*outputs), dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR TSN::SaveInferResult(std::vector<std::vector<float>> *batchFeaturePaths,
 const std::vector<MxBase::TensorBase> &inputs) {
    std::vector<float> tmp;
    LogInfo << "Infer results before postprocess:\n";
    for (auto retTensor : inputs) {
        LogInfo << "Tensor description:\n" << retTensor.GetDesc();
        std::vector<uint32_t> shape = retTensor.GetShape();
        uint32_t N = shape[0];
        uint32_t C = shape[1];
        if (!retTensor.IsHost()) {
            LogInfo << "this tensor is not in host. Now deploy it to host";
            retTensor.ToHost();
        }
        void* data = retTensor.GetBuffer();
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < C; j++) {
                float value = *(reinterpret_cast<float*>(data) + i * C + j);
                tmp.emplace_back(value);
            }
        }
    }
    (*batchFeaturePaths).emplace_back(tmp);
    return APP_ERR_OK;
}


APP_ERROR TSN::PostProcess(const std::vector<std::vector<float>> &result) {
    std::vector<float> res;
    std::string resultPathName = "./results.txt";
    std::ofstream outfile(resultPathName, std::ios::app);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    for (uint32_t i = 0; i < result[0].size(); ++i) {
        float tmp = 0;
        for (uint32_t j = 0; j < result.size(); ++j) {
            tmp += result[j][i];
        }
        res.emplace_back(tmp/result.size());
    }
    int class_id = max_element(res.begin(), res.end()) - res.begin();
    outfile << std::to_string(class_id) << std::endl;
    outfile.close();
    return APP_ERR_OK;
}


APP_ERROR TSN::GroupCenterCrop(std::vector<cv::Mat> *images, const InitParam &initParam) {
    for (auto &img : (*images)) {
        int height = img.rows;
        int width = img.cols;
        int left = (width - initParam.input_size)/2;
        int top = (height - initParam.input_size)/2;
        cv::Rect area(left, top, initParam.input_size, initParam.input_size);
        img = img(area);
    }
    return APP_ERR_OK;
}


APP_ERROR TSN::GroupScale(std::vector<cv::Mat> *images, const InitParam &initParam) {
    for (auto &img : (*images)) {
        int h = img.rows;
        int w = img.cols;
        cv::Size dsize;
        if (w > h) {
            dsize = cv::Size(static_cast<int>(initParam.scale_size *
             static_cast<float>(w) / h), initParam.scale_size);
        } else {
            dsize = cv::Size(initParam.scale_size, static_cast<int>(initParam.scale_size
             * static_cast<float>(h) / w));
        }
        cv::resize(img, img, dsize, cv::INTER_LINEAR);
    }
    return APP_ERR_OK;
}


void fill_fix_offset(std::vector<std::vector<int>> *ret, const int image_w, const int image_h,
 const int crop_w, const int crop_h) {
    //  locate crop location
    int w_step = (image_w - crop_w) / 4;
    int h_step = (image_h - crop_h) / 4;

    (*ret).push_back(std::vector<int>{0, 0});
    (*ret).push_back(std::vector<int>{4 * w_step, 0});
    (*ret).push_back(std::vector<int>{0, 4 * h_step});
    (*ret).push_back(std::vector<int>{4 * w_step, 4 * h_step});
    (*ret).push_back(std::vector<int>{2 * w_step, 2 * h_step});
}


APP_ERROR TSN::GroupOverSample(std::vector<cv::Mat> *images, const InitParam &initParam) {
    std::vector<cv::Mat> new_images;
    GroupScale(images, initParam);
    std::vector<std::vector<int>> offsets;
    int image_w = (*images)[0].cols;
    int image_h = (*images)[0].rows;

    fill_fix_offset(&offsets, image_w, image_h, initParam.input_size, initParam.input_size);
    for (auto o_wh : offsets) {
        std::vector<cv::Mat> normal_group;
        std::vector<cv::Mat> flip_group;
        cv::Rect area(o_wh[0], o_wh[1], initParam.input_size, initParam.input_size);
        for (uint32_t i = 0; i < (*images).size(); ++i) {
            cv::Mat crop = (*images)[i](area);
            normal_group.push_back(crop);
            cv::Mat flip_crop;
            crop.copyTo(flip_crop);
            cv::flip(flip_crop, flip_crop, 1);
            flip_group.push_back(flip_crop);
        }
        new_images.insert(new_images.end(), normal_group.begin(), normal_group.end());
        new_images.insert(new_images.end(), flip_group.begin(), flip_group.end());
    }
    images->clear();
    images->insert(images->end(), new_images.begin(), new_images.end());
    return APP_ERR_OK;
}


APP_ERROR TSN::GroupNormalize(std::vector<cv::Mat> *images, const InitParam &initParam) {
    //  GroupNormalize
    for (auto &img : (*images)) {
        if (initParam.modality == "Flow") {
            img.convertTo(img, CV_32F, 1 / initParam.input_std[0], - initParam.input_mean[0] / initParam.input_std[0]);
        } else {
            constexpr size_t ALPHA_AND_BETA_SIZE = 3;
            cv::Mat float32Mat;
            img.convertTo(float32Mat, CV_32FC3);
            std::vector<cv::Mat> tmp;
            cv::split(float32Mat, tmp);

            for (size_t i = 0; i < ALPHA_AND_BETA_SIZE; ++i) {
                tmp[i].convertTo(tmp[i], CV_32FC3, 1. / initParam.input_std[i],
                - initParam.input_mean[i] / initParam.input_std[i]);
            }
            cv::merge(tmp, img);
        }
    }
    return APP_ERR_OK;
}


APP_ERROR TSN::CVMatToTensorBase(float* data, MxBase::TensorBase *tensorBase, const InitParam &initParam) {
    const uint32_t dataSize = initParam.length * 224 * 224 * sizeof(float);
    LogInfo << dataSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, static_cast<uint32_t>(initParam.length), 224, 224};
    (*tensorBase) = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}


APP_ERROR TSN::Process(const std::string &dataPath, const std::string &image_tmpl,
 const std::vector<int> &indices, const InitParam &initParam) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    std::vector<std::vector<float>> result = {};
    MxBase::TensorBase tensorBase;
    std::vector<cv::Mat> images;

    for (auto frame : indices) {
        std::string path = initParam.data_path + "/" + dataPath;
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << frame;
        if (initParam.modality == "Flow") {
            std::string path_x = path + "/" + image_tmpl + "x_" + ss.str()  + ".jpg";
            std::string path_y = path + "/" + image_tmpl + "y_" + ss.str()  + ".jpg";
            images.push_back(cv::imread(path_x, cv::IMREAD_GRAYSCALE));
            images.push_back(cv::imread(path_y, cv::IMREAD_GRAYSCALE));
        } else {
            std::string path_rgb = path + "/" + image_tmpl + ss.str()  + ".jpg";
            images.push_back(cv::imread(path_rgb, cv::IMREAD_COLOR));
        }
    }
    if (initParam.test_crops == 10 && initParam.modality == "RGB") {
        GroupOverSample(&images, initParam);
    } else {
        GroupScale(&images, initParam);
        GroupCenterCrop(&images, initParam);
    }
    GroupNormalize(&images, initParam);
    uint32_t step = initParam.length;
    if (initParam.modality != "Flow") {
        step /= 3;
    }

    uint32_t dataSize = initParam.length * initParam.input_size * initParam.input_size;
    float *data = new float[dataSize];
    for (uint32_t i = 0; i < images.size(); i += step) {
        uint32_t idx = 0;
        for (uint32_t j = i; j < i + step; ++j) {
            int height = static_cast<int>(images[j].rows);
            int width = static_cast<int>(images[j].cols);
            if (initParam.modality != "Flow") {
                idx = 0;
            }
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    if (initParam.modality == "Flow") {
                        data[idx++] = images[j].at<float>(h, w);
                    } else if (initParam.modality == "RGB") {
                        data[idx] = images[j].at<cv::Vec3f>(h, w)[0];
                        data[height * width + idx] = images[j].at<cv::Vec3f>(h, w)[1];
                        data[height * width * 2 + idx] = images[j].at<cv::Vec3f>(h, w)[2];
                        ++idx;
                    }
                }
            }
        }
        CVMatToTensorBase(data, &tensorBase, initParam);
        inputs.clear();
        outputs.clear();
        inputs.push_back(tensorBase);
        auto ret = Inference(inputs, &outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }

        ret = SaveInferResult(&result, outputs);
    }
    PostProcess(result);
    return APP_ERR_OK;
}
