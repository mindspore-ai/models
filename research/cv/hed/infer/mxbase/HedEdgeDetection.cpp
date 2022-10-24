/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "HedEdgeDetection.h"

#include <memory>
#include <vector>
#include <string>
#include<fstream>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/Log/Log.h"


APP_ERROR HedEdgeDetection::Init(const InitParam &initParam) {
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
    uint32_t inputModelChannel = modelDesc_.inputTensors[0].tensorDims[MxBase::VECTOR_SECOND_INDEX];
    uint32_t inputModelHeight = modelDesc_.inputTensors[0].tensorDims[MxBase::VECTOR_THIRD_INDEX];
    uint32_t inputModelWidth = modelDesc_.inputTensors[0].tensorDims[MxBase::VECTOR_FOURTH_INDEX];
    TargetChannel_ = inputModelChannel;
    TargetHeight_ = inputModelHeight;
    TargetWidth_ = inputModelWidth;
    LogInfo << " TargetChannel_:" << TargetChannel_;
    LogInfo << " TargetHeight_:" << TargetHeight_ << " TargetWidth_:" <<TargetWidth_;
    return APP_ERR_OK;
}


APP_ERROR HedEdgeDetection::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}


int HedEdgeDetection::getBinSize(std::string path) {
    int size = 0;
    std::ifstream infile(path, std::ifstream::binary);
    infile.seekg(0, infile.end);
    size = infile.tellg();
    infile.seekg(0, infile.beg);
    infile.close();
    std::cout << "path: " <<path << " size/sizeof(float): " <<
            size/sizeof(float) << std::endl;
    return size/sizeof(float);
}

APP_ERROR HedEdgeDetection::readBin(std::string path, float *buf, int size) {
    std::ifstream infile(path, std::ifstream::binary);
    infile.read(reinterpret_cast<char*>(buf), sizeof(float)*size);
    infile.close();
    return APP_ERR_OK;
}

APP_ERROR HedEdgeDetection::writeBin(std::string path, float *buf, int size) {
    std::ofstream outfile(path, std::ifstream::binary);
    outfile.write(reinterpret_cast<char*>(buf), size*sizeof(float));
    outfile.close();
    return APP_ERR_OK;
}

APP_ERROR HedEdgeDetection::BinToTensorBase(const int length, float *image, const uint32_t target_channel,
                                            const uint32_t target_width, const uint32_t target_height,
                                            MxBase::TensorBase *tensorBase) {
    const uint32_t dataSize = static_cast<uint32_t>(length);
    MxBase::MemoryData memoryDataDst(dataSize*4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(image, dataSize*4, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, target_channel, target_height, target_width};
    *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR HedEdgeDetection::Inference(std::vector<MxBase::TensorBase> *inputs,
                                      std::vector<MxBase::TensorBase> *outputs) {
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
        outputs->push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    APP_ERROR ret = model_->ModelInference(*inputs, *outputs, dynamicInfo);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}


APP_ERROR HedEdgeDetection::PostProcess(std::vector<MxBase::TensorBase> *inputs, float *buf) {
        MxBase::TensorBase tensor = *inputs->begin();
        int ret = tensor.ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Tensor deploy to host failed.";
            return ret;
        }
        uint32_t outputModelChannel = tensor.GetShape()[MxBase::VECTOR_SECOND_INDEX];
        uint32_t outputModelHeight = tensor.GetShape()[MxBase::VECTOR_THIRD_INDEX];
        uint32_t outputModelWidth = tensor.GetShape()[MxBase::VECTOR_FOURTH_INDEX];
        LogInfo << "Channel:" << outputModelChannel << " Height:"
                << outputModelHeight << " Width:" <<outputModelWidth;
        auto data = reinterpret_cast<float(*)[outputModelChannel]
        [outputModelHeight][outputModelWidth]>(tensor.GetBuffer());
        for (size_t c = 0; c < outputModelChannel; ++c) {
            for (size_t x = 0; x < outputModelHeight; ++x) {
                for (size_t y = 0; y < outputModelWidth; ++y) {
                    size_t index = y + x * outputModelWidth +
                                   c * outputModelWidth * outputModelHeight;
                    buf[index] = data[0][c][x][y];
                }
            }
        }
        return APP_ERR_OK;
}

APP_ERROR HedEdgeDetection::Process(const std::string &imgPath, const std::string &resultPath) {
    int length = getBinSize(imgPath);
    float* image = new float[length];
    APP_ERROR ret = readBin(imgPath, image, length);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    MxBase::TensorBase tensorBase;
    ret = BinToTensorBase(length, image, TargetChannel_, TargetWidth_, TargetHeight_, &tensorBase);
    if (ret != APP_ERR_OK) {
        LogError << "BinToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(tensorBase);
    ret = Inference(&inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    float* output = new float[length/TargetChannel_];
    ret = PostProcess(&outputs, output);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    std::string imgName = imgPath;
    size_t pos_begin = imgName.find_last_of("/");
    if (static_cast<int>(pos_begin) == -1) {
        imgName = "./" + imgName;
        pos_begin = imgName.find_last_of("/");
    }
    imgName.replace(imgName.begin(), imgName.begin()+pos_begin, "");
    size_t pos_end = imgName.find_last_of(".");
    imgName.replace(imgName.begin() + pos_end, imgName.end(), ".bin");
    std::string resultPathfile = resultPath + imgName;
    LogInfo << "resultPathfile: " << resultPathfile;
    ret = writeBin(resultPathfile, output, length/TargetChannel_);
    if (ret != APP_ERR_OK) {
        LogError << "writeBin failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
