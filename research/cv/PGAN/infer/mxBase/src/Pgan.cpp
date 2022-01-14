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

#include "Pgan.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

namespace ProcessParameters {
    const float NORMALIZE_MEAN = -1.;
    const float NORMALIZE_STD = 2.;
    const float CLIP_MIN = -1.;
    const float CLIP_MAX = 1.;
    const uint32_t INPUT_DIM = 512;
    const uint32_t OUTPUT_HEIGHT_SINGLE = 128;
    const uint32_t OUTPUT_WIDTH_SINGLE = 128;
    const uint32_t CHANNEL = 3;
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

void PrintInputShape(const std::vector<MxBase::TensorBase> &input) {
    MxBase::TensorBase inputNoise = input[0];
    LogInfo << "  -------------------------input0 ";
    LogInfo << inputNoise.GetDataType();
    LogInfo << inputNoise.GetShape()[0] << ", " << inputNoise.GetShape()[1];
    LogInfo << inputNoise.GetSize();
}
void PrintOutputShape(const std::vector<MxBase::TensorBase> &output) {
    MxBase::TensorBase img = output[0];
    LogInfo << "  -------------------------output ";
    LogInfo << img.GetDataType();
    LogInfo << img.GetShape()[0] << ", " << img.GetShape()[1] << ", " << img.GetShape()[2] << ", " << img.GetShape()[3];
    LogInfo << img.GetSize();
}

std::vector<uint32_t> ComputeOutputSizes(uint32_t batchSize) {
    uint32_t size1 = (uint32_t) sqrt(batchSize);
    uint32_t size2 = size1;
    std::vector<uint32_t> sizes = {};
    if (size1 * size1 < batchSize) {
        for (uint32_t i = size1; i >= 1; i--) {
            if (batchSize % i == 0) {
                size1 = i;
                size2 = (uint32_t) batchSize / size1;
                break;
            }
        }
    }
    LogInfo << size1 << " " << size2;
    sizes.push_back(size1);
    sizes.push_back(size2);
    return sizes;
}

APP_ERROR Pgan::Init(const InitParam &initParam, uint32_t batchSize) {
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
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    srPath_ = initParam.srPath;
    PrintTensorShape(modelDesc_.inputTensors, "Model Input Tensors");
    PrintTensorShape(modelDesc_.outputTensors, "Model Output Tensors");


    return APP_ERR_OK;
}

APP_ERROR Pgan::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();

    MxBase::DeviceManager::GetInstance()->DestroyDevices();

    return APP_ERR_OK;
}

APP_ERROR Pgan::BuildNoiseData(uint32_t batchSize, uint32_t inputRound, std::vector<MxBase::TensorBase> &inputs) {
    // construct a blank input tensor.
    using ProcessParameters::INPUT_DIM;
    uint32_t dataSize = INPUT_DIM * batchSize;
    std::default_random_engine gen(time(0));
    std::normal_distribution<float> normal(0, 1);

    int rowsNum = inputRound;
    int colsNum = dataSize;

    float** mat_data = new float*[rowsNum];
    for (int i = 0; i < rowsNum; i++) {
        mat_data[i] = new float[colsNum];
    }

    for (size_t i = 0; i < inputRound; i++) {
        for (size_t b = 0; b < batchSize; b++) {
            for (size_t c = 0; c < INPUT_DIM; c++) {
                mat_data[i][c + INPUT_DIM * b] = normal(gen);
            }
        }
    }

    for (size_t i = 0; i < inputRound; i++) {
        MxBase::TensorBase tensorBase;

        float *mat_datai = mat_data[i];
        MxBase::MemoryData memoryDataDst(dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
        MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(&mat_datai[0]), dataSize * 4,
                                          MxBase::MemoryData::MEMORY_HOST_MALLOC);
        APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Memory malloc failed.";
            return ret;
        }
        std::vector <uint32_t> shape = {batchSize, INPUT_DIM};
        tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
        inputs.push_back(tensorBase);
    }
    return APP_ERR_OK;
}


APP_ERROR Pgan::Inference(const std::vector<MxBase::TensorBase> &inputs,
                          std::vector<MxBase::TensorBase> &outputs, uint32_t inputBatch) {
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
    dynamicInfo.batchSize = inputBatch;

    PrintInputShape(inputs);

    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);

    PrintOutputShape(outputs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR Pgan::PostProcess(std::vector<MxBase::TensorBase> &outputs, cv::Mat &resultImg,
                            uint32_t size1, uint32_t size2) {
    using ProcessParameters::NORMALIZE_MEAN;
    using ProcessParameters::NORMALIZE_STD;
    using ProcessParameters::CLIP_MIN;
    using ProcessParameters::CLIP_MAX;
    using ProcessParameters::OUTPUT_HEIGHT_SINGLE;
    using ProcessParameters::OUTPUT_WIDTH_SINGLE;
    using ProcessParameters::CHANNEL;
    LogInfo << "output_size:" << outputs.size();
    LogInfo <<  "output0_datatype:" << outputs[0].GetDataType();
    LogInfo << "output0_shape:" << outputs[0].GetShape()[0] << ", " << outputs[0].GetShape()[1] << ", "
            << outputs[0].GetShape()[2] << ", " << outputs[0].GetShape()[3];
    LogInfo << "output0_bytesize:"  << outputs[0].GetByteSize();

    size_t batch = modelDesc_.outputTensors[0].tensorDims[0];

    APP_ERROR ret = outputs[0].ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "tohost fail.";
        return ret;
    }

    float *outputPtr =  reinterpret_cast<float *>(outputs[0].GetBuffer());

    size_t  H = OUTPUT_HEIGHT_SINGLE, W = OUTPUT_WIDTH_SINGLE, C = CHANNEL,
            org_H = resultImg.rows, org_W = resultImg.cols;

    cv::Mat outputImg(H*size1, W*size2, CV_8UC3);

    float tmpNum;
    size_t i = 0, tmpSizeH = 0, tmpSizeW = 0, tmpImages = 0;

    for (size_t b1 = 0; b1 < size_t(size1); b1++) {
        for (size_t b2 = 0; b2 < size_t(size2); b2++) {
            if (((tmpImages % batch) == 0) && (i < (outputs.size()))) {
                ret = outputs[i].ToHost();
                if (ret != APP_ERR_OK) {
                    LogError << GetError(ret) << "tohost fail.";
                    return ret;
                }
                outputPtr =  reinterpret_cast<float *>(outputs[i].GetBuffer());
                tmpSizeH = 0;
                tmpSizeW = 0;
                i++;
            }
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        tmpNum = *(outputPtr + (C - c - 1) * (H * W) + h * W + w +
                                   (tmpSizeH * static_cast<int>(size2) * H * W * C + tmpSizeW * H * W * C));
                        // clip to [-1, 1]
                        if (tmpNum < CLIP_MIN) {
                            tmpNum = CLIP_MIN;
                        } else if (tmpNum > CLIP_MAX) {
                            tmpNum = CLIP_MAX;
                        }
                        tmpNum = ((tmpNum - NORMALIZE_MEAN) / NORMALIZE_STD) * 255;
                        outputImg.at<cv::Vec3b>(h + H * b1, w + W * b2)[c] = static_cast<int>(tmpNum);
                    }
                }
            }
            tmpSizeW++;
            if (tmpSizeW == (size_t)size2) {
                tmpSizeH++;
                tmpSizeW = 0;
            }
            tmpImages++;
        }
    }
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < org_H; h++) {
            for (size_t w = 0; w < org_W; w++) {
                resultImg.at<cv::Vec3b>(h, w)[c] = outputImg.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR Pgan::SaveResult(cv::Mat &resultImg, const std::string imgName) {
    DIR *dirPtr = opendir(srPath_.c_str());
    if (dirPtr == nullptr) {
        std::string path1 = "mkdir -p " + srPath_;
        system(path1.c_str());
    }
    cv::imwrite(srPath_ + "/" + imgName, resultImg);
    return APP_ERR_OK;
}

APP_ERROR Pgan::Process(uint32_t batchSize) {
    APP_ERROR ret;

    uint32_t inputRound = (uint32_t) batchSize / (uint32_t)modelDesc_.inputTensors[0].tensorDims[0];
    if ((uint32_t) batchSize % (uint32_t)modelDesc_.inputTensors[0].tensorDims[0] != 0) {
        inputRound++;
    }

    std::vector<MxBase::TensorBase> outputs = {};
    std::vector<MxBase::TensorBase> inputs = {};
    ret = BuildNoiseData((uint32_t)modelDesc_.inputTensors[0].tensorDims[0], inputRound, inputs);
    if (ret != APP_ERR_OK) {
        LogError << "BuildNoiseData failed, ret=" << ret << ".";
        return ret;
    }

    for (size_t i = 0; i < inputRound; i++) {
        std::vector<MxBase::TensorBase> outputs_tmp = {};
        std::vector<MxBase::TensorBase> inputs_tmp = {};
        MxBase::TensorBase tensorBase0 = inputs[i];
        inputs_tmp.push_back(tensorBase0);
        auto startTime = std::chrono::high_resolution_clock::now();
        ret = Inference(inputs_tmp, outputs_tmp, inputRound);
        auto endTime = std::chrono::high_resolution_clock::now();
        double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
        inferCostTimeMilliSec += costMs;
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(outputs_tmp[0]);
    }

    using ProcessParameters::OUTPUT_HEIGHT_SINGLE;
    using ProcessParameters::OUTPUT_WIDTH_SINGLE;
    std::vector<uint32_t> sizes = ComputeOutputSizes(batchSize);
    cv::Mat resultImg(OUTPUT_HEIGHT_SINGLE*sizes[0], OUTPUT_WIDTH_SINGLE*sizes[1], CV_8UC3);
    ret = PostProcess(outputs, resultImg, sizes[0], sizes[1]);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgName = "result.jpg";
    ret = SaveResult(resultImg, imgName);
    if (ret != APP_ERR_OK) {
        LogError << "Save infer results into file failed. ret = " << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}
