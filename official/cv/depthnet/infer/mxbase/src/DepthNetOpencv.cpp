/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd.
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

#include "DepthNetOpencv.h"
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR DepthNetOpencv::Init(const InitParam& initParam) {
    this->deviceId_ = initParam.deviceId;
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
    this->coarseModel_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = this->coarseModel_->Init(initParam.CoarseModelPath, this->coarseModelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "CoarseModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    this->fineModel_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = this->fineModel_->Init(initParam.FineModelPath, this->fineModelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "FineModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    uint32_t rgb_input_data_size = 1;
    for (size_t j = 0; j < this->coarseModelDesc_.inputTensors[0].tensorDims.size(); ++j) {
        this->rgbInputDataShape_[j] = (uint32_t)this->coarseModelDesc_.inputTensors[0].tensorDims[j];
        rgb_input_data_size *= this->rgbInputDataShape_[j];
    }
    uint32_t depth_input_data_size = 1;
    for (size_t j = 0; j < this->fineModelDesc_.inputTensors[0].tensorDims.size(); ++j) {
        this->depthInputDataShape_[j] = (uint32_t)this->fineModelDesc_.inputTensors[0].tensorDims[j];
        depth_input_data_size *= this->depthInputDataShape_[j];
    }

    this->rgbInputDataSize_ = rgb_input_data_size;
    this->depthInputDataSize_ = depth_input_data_size;
    std::cout << "rgb_input_data_size: " << this->rgbInputDataSize_ << std::endl;
    std::cout << "depth_input_data_size: " << this->depthInputDataSize_ << std::endl;

    return APP_ERR_OK;
}

APP_ERROR DepthNetOpencv::DeInit() {
    this->coarseModel_->DeInit();
    this->fineModel_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR DepthNetOpencv::ReadTensorFromFile(const std::string& file, float* data, const int flag) {
    if (data == NULL) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    std::ifstream infile;
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check data file validity
    if (infile.fail()) {
        LogError << "Failed to open data file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    if (flag == 1) infile.read(reinterpret_cast<char*>(data), sizeof(float) * this->rgbInputDataSize_);
    else
        infile.read(reinterpret_cast<char*>(data), sizeof(float) * this->depthInputDataSize_);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR DepthNetOpencv::ReadInputTensor(const std::string& fileName,
    std::vector<MxBase::TensorBase>* inputs, const int flag) {
    if (flag == 1) {
        float* data = new float[this->rgbInputDataSize_];
        APP_ERROR ret = ReadTensorFromFile(fileName, data, flag);
        if (ret != APP_ERR_OK) {
            LogError << "Coarse - ReadTensorFromFile failed.";
            return ret;
        }
        const uint32_t dataSize = this->coarseModelDesc_.inputTensors[0].tensorSize;
        std::cout << "rgbInputDataSize_:" << dataSize << std::endl;
        MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, this->deviceId_);
        MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize,
            MxBase::MemoryData::MEMORY_HOST_MALLOC);

        ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Memory malloc and copy failed.";
            return ret;
        }
        inputs->push_back(MxBase::TensorBase(memoryDataDst, false,
            this->rgbInputDataShape_, MxBase::TENSOR_DTYPE_FLOAT32));
    } else {
        float* data = new float[this->depthInputDataSize_];
        APP_ERROR ret = ReadTensorFromFile(fileName, data, flag);
        if (ret != APP_ERR_OK) {
            LogError << "Fine - ReadTensorFromFile failed.";
            return ret;
        }
        const uint32_t dataSize = this->fineModelDesc_.inputTensors[1].tensorSize;
        std::cout << "depthInputDataSize_:" << dataSize << std::endl;
        MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, this->deviceId_);
        MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize,
            MxBase::MemoryData::MEMORY_HOST_MALLOC);

        ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Memory malloc and copy failed.";
            return ret;
        }
        inputs->push_back(MxBase::TensorBase(memoryDataDst, false,
            this->depthInputDataShape_, MxBase::TENSOR_DTYPE_FLOAT32));
    }

    return APP_ERR_OK;
}

APP_ERROR DepthNetOpencv::Inference(const std::vector<MxBase::TensorBase>& inputs,
    std::vector<MxBase::TensorBase>* outputs, const int flag) {
    if (flag == 1) {
        auto dtypes = this->coarseModel_->GetOutputDataType();
        for (size_t i = 0; i < this->coarseModelDesc_.outputTensors.size(); ++i) {
            std::vector<uint32_t> shape = {};
            for (size_t j = 0; j < coarseModelDesc_.outputTensors[i].tensorDims.size(); ++j) {
                shape.push_back((uint32_t)this->coarseModelDesc_.outputTensors[i].tensorDims[j]);
            }
            MxBase::TensorBase tensor(shape, dtypes[i],
                MxBase::MemoryData::MemoryType::MEMORY_DEVICE, this->deviceId_);
            APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
            if (ret != APP_ERR_OK) {
                LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
                return ret;
            }
            outputs->push_back(tensor);

            MxBase::DynamicInfo dynamicInfo = {};
            dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
            auto startTime = std::chrono::high_resolution_clock::now();

            // coarse infer
            ret = this->coarseModel_->ModelInference(inputs, *outputs, dynamicInfo);
            if (ret != APP_ERR_OK) {
                LogError << "CoarseModelInference failed, ret=" << ret << ".";
                return ret;
            }
            auto endTime = std::chrono::high_resolution_clock::now();
            // save time
            double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            inferCostTimeMilliSec += costMs;
        }
    } else {
        auto dtypes = this->fineModel_->GetOutputDataType();
        for (size_t i = 0; i < this->fineModelDesc_.outputTensors.size(); ++i) {
            std::vector<uint32_t> shape = {};
            for (size_t j = 0; j < fineModelDesc_.outputTensors[i].tensorDims.size(); ++j) {
                shape.push_back((uint32_t)this->fineModelDesc_.outputTensors[i].tensorDims[j]);
            }
            MxBase::TensorBase tensor(shape, dtypes[i],
                MxBase::MemoryData::MemoryType::MEMORY_DEVICE, this->deviceId_);
            APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
            if (ret != APP_ERR_OK) {
                LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
                return ret;
            }
            outputs->push_back(tensor);

            MxBase::DynamicInfo dynamicInfo = {};
            dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
            auto startTime = std::chrono::high_resolution_clock::now();
            // fine infer
            ret = this->fineModel_->ModelInference(inputs, *outputs, dynamicInfo);
            if (ret != APP_ERR_OK) {
                LogError << "CoarseModelInference failed, ret=" << ret << ".";
                return ret;
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            // save time
            double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            inferCostTimeMilliSec += costMs;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR DepthNetOpencv::SaveResult(const std::string& result,
    std::vector<MxBase::TensorBase> outputs) {
    for (size_t i = 0; i < outputs.size(); ++i) {
        APP_ERROR ret = outputs[i].ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "to host fail.";
            return ret;
        }
        auto* netOutput = reinterpret_cast<float*>(outputs[i].GetBuffer());
        std::vector<uint32_t> out_shape = outputs[i].GetShape();

        // create bin and write
        FILE* outputFile_ = fopen(result.c_str(), "wb");
        fwrite(netOutput, out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3],
            sizeof(float), outputFile_);
        fclose(outputFile_);
    }
    return APP_ERR_OK;
}

APP_ERROR DepthNetOpencv::CoarseProcess(const std::string& inferPath,
    const std::string& RgbBinFile) {
    // flag = 1 -> coarse_infer
    // read rgb_bin
    uint32_t flag = 1;
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputBinFile = inferPath + RgbBinFile;
    APP_ERROR ret = ReadInputTensor(inputBinFile, &inputs, flag);

    if (ret != APP_ERR_OK) {
        LogError << "Coarse - Read inputrgbbinfile failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> outputs = {};
    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs, flag);
    auto endTime = std::chrono::high_resolution_clock::now();
    // save time
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Coarse - Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::string finalRetPath = "./coarse_infer_result/" + RgbBinFile;
    finalRetPath.replace(finalRetPath.find("colors.bin"), 16, "coarse_depth.bin");
    ret = SaveResult(finalRetPath, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Coarse - SaveResult failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DepthNetOpencv::FineProcess(const std::string& inferPath, const std::string& RgbBinFile,
    const std::string& CoarseDepthBinFile) {
    std::vector<MxBase::TensorBase> inputs = {};
    // flag = 1 -> 2 -> fine_infer
    // read rgb_bin
    uint32_t flag = 1;
    std::string inputRgbBinFile = inferPath + RgbBinFile;
    APP_ERROR ret = ReadInputTensor(inputRgbBinFile, &inputs, flag);

    if (ret != APP_ERR_OK) {
        LogError << "Fine - Read inputrgbbinfile failed, ret=" << ret << ".";
        return ret;
    }

    // read depth_bin
    flag = 2;
    std::string inputDepthBinFile = CoarseDepthBinFile;
    ret = ReadInputTensor(inputDepthBinFile, &inputs, flag);

    if (ret != APP_ERR_OK) {
        LogError << "Fine - Read inputdepthbinfile failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> outputs = {};
    auto startTime = std::chrono::high_resolution_clock::now();
    ret = Inference(inputs, &outputs, flag);
    auto endTime = std::chrono::high_resolution_clock::now();
    // save time
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMilliSec += costMs;
    if (ret != APP_ERR_OK) {
        LogError << "Fine - Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::string finalRetPath = "./fine_infer_result/" + RgbBinFile;
    finalRetPath.replace(finalRetPath.find("colors.bin"), 14, "fine_depth.bin");
    ret = SaveResult(finalRetPath, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Fine - SaveResult failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
