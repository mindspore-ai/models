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
#include "TextrcnnBase.h"
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
#include <map>
#include <fstream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

const uint32_t EACH_LABEL_LENGTH = 1;
const uint32_t MAX_LENGTH = 50;
const uint32_t CLASS_NUM = 2;

/*
Load the label from label path.
*/
APP_ERROR TextrcnnBase::load_labels(const std::string &labelPath, std::vector<std::string> *labelMap) {
    std::ifstream infile;
    // open label file
    infile.open(labelPath, std::ios_base::in);
    std::string s;
    // check label file validity
    if (infile.fail()) {
        LogError << "Failed to open label file: " << labelPath << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    labelMap->clear();
    // construct label vector
    while (std::getline(infile, s)) {
        if (s.size() == 0 || s[0] == '#') {
            continue;
        }
        size_t eraseIndex = s.find_last_not_of("\r\n\t");
        if (eraseIndex != std::string::npos) {
            s.erase(eraseIndex + 1, s.size() - eraseIndex);
        }
        labelMap->push_back(s);
    }
    infile.close();
    return APP_ERR_OK;
}

/*
Constructor
*/
APP_ERROR TextrcnnBase::init(const InitParam &initParam) {
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
    classNum_ = initParam.classNum;
    // load labels from file
    ret = load_labels(initParam.labelPath, &labelMap_);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to load labels, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/*
Destructor
*/
APP_ERROR TextrcnnBase::de_init() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

/*
Read Tensor from input binary file.
*/
APP_ERROR TextrcnnBase::read_tensor_from_file(const std::string &file, uint32_t *data, uint32_t size) {
    if (data == NULL || size < MAX_LENGTH) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    std::ifstream infile;
    // open label file
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check label file validity
    if (infile.fail()) {
        LogError << "Failed to open label file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    infile.read(reinterpret_cast<char*>(data), sizeof(uint32_t) * MAX_LENGTH);
    infile.close();
    return APP_ERR_OK;
}

/*
Generate tensor for input into model.
*/
APP_ERROR TextrcnnBase::read_input_tensor(const std::string &fileName, uint32_t index,
                                       std::vector<MxBase::TensorBase> *inputs) {
    uint32_t data[MAX_LENGTH] = {0};
    APP_ERROR ret = read_tensor_from_file(fileName, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "read_tensor_from_file failed.";
        return ret;
    }

    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, MAX_LENGTH};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT32));
    return APP_ERR_OK;
}

/*
Get the input tensor, infer the output tensor.
*/
APP_ERROR TextrcnnBase::inference(const std::vector<MxBase::TensorBase> &inputs,
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
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_infer_cost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/*
Transform the output tensor into category label.
*/
APP_ERROR TextrcnnBase::post_process(std::vector<MxBase::TensorBase> *outputs, std::vector<uint32_t> *argmax) {
    MxBase::TensorBase &tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }

    // // check tensor is available, print the output shape of model.
    // auto outputShape = tensor.GetShape();
    // uint32_t length = outputShape[0];
    // uint32_t classNum = outputShape[1];
    // LogInfo << "output shape is: " << outputShape[0] << " "<< outputShape[1] << std::endl;

    void* data = tensor.GetBuffer();

    // the type of model output should be float16, 'short' is used here, the size relationship remains the same
    int16_t value0 = *(reinterpret_cast<int16_t*>(data));
    int16_t value1 = *(reinterpret_cast<int16_t*>(data) + 1);

    // LogInfo << "[ " << value0 << " " << value1 << " ] ";
    uint32_t argmaxIndex = value1 > value0 ? 1 : 0;
    argmax->push_back(argmaxIndex);

    return APP_ERR_OK;
}

/*
Count the predict results.
*/
APP_ERROR TextrcnnBase::count_predict_result(const std::string &labelFile, const std::vector<uint32_t> &argmax) {
    uint32_t data[MAX_LENGTH] = {0};
    APP_ERROR ret = read_tensor_from_file(labelFile, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogInfo << "count_predict_result";
        LogError << "read_tensor_from_file failed.";
        return ret;
    }
    if (argmax[0] == 1) {  // Prediction
        if (data[0] == 1) {  // Fact label
            g_tp += 1;
        } else {
            g_fp += 1;
        }
    } else {
        if (data[0] == 1) {
            g_fn += 1;
        } else {
            g_tn += 1;
        }
    }

    LogInfo << "TP: " << g_tp << ", FP: " << g_fp << ", FN: " << g_fn << ", TN: " << g_tn;
    return APP_ERR_OK;
}

/*
Append the infer result to terminal and file: "../result/result.txt".
*/
APP_ERROR TextrcnnBase::write_result(const std::string &fileName, const std::vector<uint32_t> &argmax) {
    std::string resultPathName = "result";
    // create result directory when it does not exit
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    // create result file under result directory
    resultPathName = resultPathName + "/result.txt";
    std::ofstream tfile(resultPathName, std::ofstream::app);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // write inference result into file
    LogInfo << "infer result of " << fileName << " is: " << labelMap_[argmax[0]];
    tfile << "infer result of " << fileName << " is: " << labelMap_[argmax[0]] << std::endl;
    tfile.close();
    return APP_ERR_OK;
}

/*
The main function that call other functions to complete the whole process of one binary file.
*/
APP_ERROR TextrcnnBase::process(const std::string &inferPath, const std::string &fileName, bool eval) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputIdsFile = inferPath + "/00_feature/" + fileName;
    APP_ERROR ret = read_input_tensor(inputIdsFile, INPUT_IDS, &inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input ids failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<MxBase::TensorBase> outputs = {};
    ret = inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<uint32_t> argmax;
    ret = post_process(&outputs, &argmax);
    if (ret != APP_ERR_OK) {
        LogError << "post_process failed, ret=" << ret << ".";
        return ret;
    }

    ret = write_result(fileName, argmax);
    if (ret != APP_ERR_OK) {
        LogError << "save result failed, ret=" << ret << ".";
        return ret;
    }

    if (eval) {
        std::string labelFile = inferPath + "/labels/" + fileName;
        ret = count_predict_result(labelFile, argmax);
        if (ret != APP_ERR_OK) {
            LogError << "CalcF1Score read label failed, ret=" << ret << ".";
            return ret;
        }
    }

    LogInfo << "==============================================================";

    return APP_ERR_OK;
}
