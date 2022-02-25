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

#include <sys/stat.h>
#include <warpctc.h>
#include <unistd.h>
#include <memory>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "acl/acl.h"


union Fp32{
    uint32_t u;
    float f;
};

uint16_t float_cov_uint16(float value) {
    const Fp32 f32infty = { 255U << 23 };
    const Fp32 f16infty = { 31U << 23 };
    const Fp32 magic = { 15U << 23 };
    const uint32_t sign_mask = 0x80000000U;
    const uint32_t round_mask = ~0xFFFU;

    Fp32 in;
    uint16_t out;
    in.f = value;
    uint32_t sign = in.u & sign_mask;
    in.u ^= sign;
    if (in.u >= f32infty.u) {  /* Inf or NaN (all exponent bits set) */
        /* NaN->sNaN and Inf->Inf */
        out = (in.u > f32infty.u) ? 0x7FFFU : 0x7C00U;
    } else {   /* (De)normalized number or zero */
        in.u &= round_mask;
        in.f *= magic.f;
        in.u -= round_mask;
        if (in.u > f16infty.u) {
            in.u = f16infty.u; /* Clamp to signed infinity if overflowed */
        }

        out = uint16_t(in.u >> 13); /* Take the bits! */
    }
    out = uint16_t(out | (sign >> 16));
    return out;
}

float uint16_cov_float(uint16_t value) {
    const Fp32 magic = { (254U - 15U) << 23 };
    const Fp32 was_infnan = { (127U + 16U) << 23 };
    Fp32 out;
    out.u = (value & 0x7FFFU) << 13;   /* exponent/mantissa bits */
    out.f *= magic.f;                  /* exponent adjust */
    if (out.f >= was_infnan.f) {        /* make sure Inf/NaN survive */
        out.u |= 255U << 23;
    }
    out.u |= (value & 0x8000U) << 16;  /* sign bit */
    return out.f;
}

// init
APP_ERROR warpCTC::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    resize_w = initParam.resize_w;
    resize_h = initParam.resize_h;

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
// DeInit
APP_ERROR warpCTC::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}
// ReadImage
APP_ERROR warpCTC::ReadImage(const std::string &imgPath, cv::Mat &imageMat) {
    cv::Mat bgrImageMat;
    bgrImageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    cv::cvtColor(bgrImageMat, imageMat, cv::COLOR_BGR2RGB);
    LogInfo << "ImageSize:" << imageMat.size();
    LogInfo << "ImageDims:" << imageMat.dims;
    LogInfo << "ImageChannels:" << imageMat.channels();
    return APP_ERR_OK;
}
// ImagePreprocess
APP_ERROR warpCTC::ImagePreprocess(cv::Mat &preimageMat, cv::Mat &DstImageMat) {
    cv::Mat pre00imageMat;
    preimageMat.convertTo(pre00imageMat, CV_32FC3);
    preimageMat.convertTo(DstImageMat, CV_16UC3);
     // rows(height) of the image
    int iRows = DstImageMat.rows;
     // cols(width) of the image
    int iCols = DstImageMat.cols;
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            float tmp0 = pre00imageMat.at<cv::Vec3f>(i, j)[0];
            float tmp00 = (tmp0/255.0-0.9010) / 0.1521;
            DstImageMat.at<cv::Vec3w>(i, j)[0] = float_cov_uint16(tmp00);

            float tmp1 = pre00imageMat.at<cv::Vec3f>(i, j)[1];
            float tmp11 = (tmp1/255.0-0.9049) / 0.1347;
            DstImageMat.at<cv::Vec3w>(i, j)[1] = float_cov_uint16(tmp11);

            float tmp2 = pre00imageMat.at<cv::Vec3f>(i, j)[2];
            float tmp22 = (tmp2/255.0-0.9025) / 0.1458;
            DstImageMat.at<cv::Vec3w>(i, j)[2] = float_cov_uint16(tmp22);
        }
    }

    return APP_ERR_OK;
}
// resize
void warpCTC::Resize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat) {
    uint32_t resizeHeight = resize_h;
    uint32_t resizeWidth = resize_w;
    cv::resize(srcImageMat, dstImageMat, cv::Size(resizeWidth, resizeHeight));
}
// CVMatToTensorBase
APP_ERROR warpCTC::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase) {
    const uint32_t dataSize =  imageMat.cols *  imageMat.rows * MxBase::YUV444_RGB_WIDTH_NU*2;
    LogInfo << "image size after resize:" << imageMat.cols << " *" << imageMat.rows;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    auto dtypes = model_->GetInputDataType();
    for (size_t i = 0; i < dtypes.size(); i++) {
        std::cout << "input dtypes:" << dtypes[i] << std::endl;
    }
    std::vector<uint32_t> shape = {1, static_cast<uint32_t>(imageMat.rows),
                                           static_cast<uint32_t>(imageMat.cols), MxBase::YUV444_RGB_WIDTH_NU };

    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT16);
    return APP_ERR_OK;
}

// model infer
APP_ERROR warpCTC::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                           std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        std::cout << "tensor info" << tensor.GetDesc() << std::endl;
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);     // request memory
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        // Put tensor into store
        outputs.push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    // Set the type to Static Batch
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR warpCTC::PostProcess(std::vector<MxBase::TensorBase> &PreInfer, std::vector<int> &output) {
    aclFloat16* chr = nullptr;
    std::vector<int> idVec;
    for (int i=0; i < PreInfer.size(); i++) {
        PreInfer[i].ToHost();
        chr = reinterpret_cast<aclFloat16*>(PreInfer[i].GetBuffer());
        for (int j = 0; j < 160; j++) {
            std::vector<float> rowVec;
            for (int k = 0; k < 11; k++) {
                rowVec.push_back(uint16_cov_float(chr[11*j+k]));
            }
            std::vector<float>::iterator maxElement = std::max_element(std::begin(rowVec), std::end(rowVec));
            uint32_t argmaxIndex = maxElement - std::begin(rowVec);
            idVec.push_back(argmaxIndex);
        }
     }
    int last_idx = 10;
    for (int i = 0; i < idVec.size(); i++) {
        if (idVec[i] != last_idx && idVec[i] != 10) {
            output.push_back(idVec[i]);
        }
        last_idx = idVec[i];
    }
    return APP_ERR_OK;
}

APP_ERROR warpCTC::Process(const std::string &imgPath) {
    cv::Mat imageMat;
    APP_ERROR ret = ReadImage(imgPath, imageMat);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    cv::Mat DstImageMat;
    ImagePreprocess(imageMat, DstImageMat);

    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> inferoutputs = {};
    MxBase::TensorBase tensorBase;
    ret = CVMatToTensorBase(DstImageMat, tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase);

    std::cout << "tensorBase info" << tensorBase.GetDesc() << std::endl;
    ret = Inference(inputs, inferoutputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<int> Output;
    ret = PostProcess(inferoutputs, Output);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    for (int i = 0; i < Output.size(); i++) {
        std::cout << "output of inference: " << Output[i] << std::endl;
    }
    return APP_ERR_OK;
}

