/*
 * Copyright 2021. Huawei Technologies Co., Ltd.
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

#include "Textfusenet.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <utility>
#include <boost/property_tree/json_parser.hpp>
#include "acl/acl.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
namespace {
const uint32_t MODEL_HEIGHT = 768;
const uint32_t MODEL_WIDTH = 1280;
}
void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";
        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

APP_ERROR Textfusenet::Init(const InitParam &initParam) {
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

    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");

    MxBase::ConfigData configData;
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("SCORE_THRESH", std::to_string(initParam.score_thresh));
    configData.SetJsonValue("IOU_THRESH", std::to_string(initParam.iou_thresh));
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    post_ = std::make_shared<MxBase::TextfusenetMindsporePost>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Textfusenet init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Textfusenet::DeInit() {
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR Textfusenet::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    return APP_ERR_OK;
}

APP_ERROR Textfusenet::Resize(cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    static constexpr uint32_t resizeHeight = 768;
    static constexpr uint32_t resizeWidth = 1280;
    int w = srcImageMat.cols;
    int h = srcImageMat.rows;
    float scale_w = resizeWidth * 1.0 / w;
    float scale_h = resizeHeight * 1.0 / h;
    float ratio = scale_w;
    if (ratio > scale_h) {
        ratio = scale_h;
    }
    int new_w = static_cast<int>(w * ratio+0.5);
    int new_h = static_cast<int>(h * ratio+0.5);
    int right = resizeWidth - new_w;
    int bottom = resizeHeight - new_h;
    cv::resize(srcImageMat, dstImageMat, cv::Size(new_w, new_h));
    cv::copyMakeBorder(dstImageMat, dstImageMat, 0, bottom, 0, right, cv::BORDER_CONSTANT, 0);
    return APP_ERR_OK;
}

APP_ERROR Textfusenet::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                 std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    dynamicInfo.batchSize = 1;

    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Textfusenet::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize = imageMat.cols * imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imageMat.cols)};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT8);
    return APP_ERR_OK;
}

APP_ERROR Textfusenet::PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                                std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
                                const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos,
                                const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
    APP_ERROR ret = post_->Process(inputs, objectInfos, resizedImageInfos, configParamMap);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Textfusenet::GetImageMeta(const ImageShape &imageShape, MxBase::TensorBase &imgMetas) const {
    float widthScale = 0;
    float heightScale = 0;

    widthScale = static_cast<float>(MODEL_WIDTH) / static_cast<float>(imageShape.width);
    if (widthScale > static_cast<float>(MODEL_HEIGHT) / static_cast<float>(imageShape.height)) {
        widthScale = static_cast<float>(MODEL_HEIGHT) / static_cast<float>(imageShape.height);
    }
    heightScale = widthScale;

    size_t metaSize = sizeof(aclFloat16) * 4;
    auto *im_info = reinterpret_cast<aclFloat16 *>(malloc(metaSize));
    im_info[0] = aclFloatToFloat16(static_cast<float>(imageShape.height));
    im_info[1] = aclFloatToFloat16(static_cast<float>(imageShape.width));
    im_info[2] = aclFloatToFloat16(heightScale);
    im_info[3] = aclFloatToFloat16(widthScale);
    void *imInfo_dst = nullptr;
    APP_ERROR ret = aclrtMalloc(&imInfo_dst, metaSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        std::cout << "aclrtMalloc failed, ret = " << ret << std::endl;
        aclrtFree(imInfo_dst);
        free(im_info);
        return ret;
    }

    ret = aclrtMemcpy(reinterpret_cast<uint8_t *>(imInfo_dst), metaSize, im_info, metaSize, ACL_MEMCPY_HOST_TO_DEVICE);
    free(im_info);
    if (ret != ACL_ERROR_NONE) {
        std::cout << "aclrtMemcpy failed, ret = " << ret << std::endl;
        aclrtFree(imInfo_dst);
        return ret;
    }

    MxBase::MemoryData memoryData(imInfo_dst, metaSize, MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
    memoryData.free = aclrtFree;

    const std::vector<uint32_t> shape = {1, 4};
    imgMetas = MxBase::TensorBase(memoryData, false, shape, MxBase::TENSOR_DTYPE_FLOAT16);
    LogInfo << "ImgMetas: " << imgMetas.GetByteSize();
    return APP_ERR_OK;
}

void SaveInferResult(const std::vector<MxBase::ObjectInfo> &objInfos, const std::string &resultPath) {
    if (objInfos.empty()) {
        LogWarn << "The predict result is empty.";
        return;
    }

    namespace pt = boost::property_tree;
    pt::ptree root, data;
    int index = 0;
    for (auto &obj : objInfos) {
        ++index;
        pt::ptree item;
        item.put("classId", obj.classId);
        item.put("className", obj.className);
        item.put("confidence", obj.confidence);
        item.put("x0", obj.x0);
        item.put("y0", obj.y0);
        item.put("x1", obj.x1);
        item.put("y1", obj.y1);

        std::string maskStr;
        for (auto &line : obj.mask) {
            for (auto &w : line) {
                std::string w_str = w ? "1" : "0";
                maskStr.append(w_str);
            }
        }
        item.put("mask", maskStr);
        item.put("maskHeight", obj.mask.size());
        auto maskWidth = obj.mask.empty() ? 0 : obj.mask[0].size();
        item.put("maskWidth", maskWidth);
        data.push_back(std::make_pair("", item));
    }
    root.add_child("data", data);
    pt::json_parser::write_json(resultPath, root, std::locale(), false);
}

APP_ERROR Textfusenet::Process(const std::string &imgPath, const std::string &resultPath) {
    ImageShape imageShape{};
    // process image
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    imageShape.width = imageMat.rows;
    imageShape.height = imageMat.cols;
    ret = Resize(imageMat, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(imageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }

    inputs.push_back(tensorBase);

    MxBase::TensorBase imgMetas;
    ret = GetImageMeta(imageShape, imgMetas);
    if (ret != APP_ERR_OK) {
        LogError << "Get Image metas failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(imgMetas);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "Inference success, imgInforet=" << ret << ".";
    std::vector<MxBase::ResizedImageInfo> resizedImageInfos = {};
    MxBase::ResizedImageInfo imgInfo = {
      MODEL_WIDTH, MODEL_HEIGHT, imageShape.width, imageShape.height, MxBase::RESIZER_STRETCHING, 0.0};
    resizedImageInfos.push_back(imgInfo);
    std::vector<std::vector<MxBase::ObjectInfo>> objectInfos = {};
    std::map<std::string, std::shared_ptr<void>> configParamMap = {};

    ret = PostProcess(outputs, objectInfos, resizedImageInfos, configParamMap);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    if (objectInfos.empty()) {
        LogInfo << "No object detected." << std::endl;
        return APP_ERR_OK;
    }

    std::vector<MxBase::ObjectInfo> objects = objectInfos.at(0);
    SaveInferResult(objects, resultPath);
    return APP_ERR_OK;
}
