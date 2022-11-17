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

#include "SentimentNet.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include <cmath>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

const uint32_t MAX_LENGTH = 128000;

/**
 * Init SentimentNet with {@link InitParam}
 * @param initParam const reference to initial param
 * @return status code of whether initialization is successful
 */
APP_ERROR SentimentNet::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    // init device manager
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    // set tensor context
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    // init model inference processor
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/**
 * De-init SentimentNet
 * @return status code of whether de-initialization is successful
 */
APP_ERROR SentimentNet::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

/**
 * read tensor from file
 * @param file const reference to file name
 * @param data the pointer to data which store the read data
 * @param size data size
 * @return status code of whether read data is successful
 */
APP_ERROR SentimentNet::ReadTensorFromFile(const std::string &file, uint32_t *data, const uint32_t size) {
    if (data == NULL || size < MAX_LENGTH) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    std::ifstream infile;
    // open sentence file
    infile.open(file, std::ios_base::in | std::ios_base::binary);
    // check sentence file validity
    if (infile.fail()) {
        LogError << "Failed to open label file: " << file << ".";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // read sentence data
    infile.read(reinterpret_cast<char *>(data), sizeof(uint32_t) * MAX_LENGTH);
    infile.close();
    return APP_ERR_OK;
}

/**
 * read input tensor from file
 * @param fileName const reference to file name
 * @param index index of modelDesc inputTensors
 * @param inputs reference to input tensor stored
 * @return status code of whether reading tensor is successful
 */
APP_ERROR SentimentNet::ReadInputTensor(const std::string &fileName, uint32_t index,
                                        const std::shared_ptr<std::vector<MxBase::TensorBase>> &inputs) {
    // read data from file
    uint32_t data[MAX_LENGTH] = {0};
    APP_ERROR ret = ReadTensorFromFile(fileName, data, MAX_LENGTH);
    if (ret != APP_ERR_OK) {
        LogError << "ReadTensorFromFile failed.";
        return ret;
    }

    // convert memory type
    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;  // 128000 64 * 2000
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }

    // construct input tensor
    std::vector <uint32_t> shape = {1, MAX_LENGTH};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_UINT32));
    return APP_ERR_OK;
}

/**
 * Sentence Classification
 * @param inputs const reference to word vector of sentence
 * @param outputs reference to the model output tensors
 * @return status code of whether inference is successful
 */
APP_ERROR SentimentNet::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                  const std::shared_ptr<std::vector<MxBase::TensorBase>> &outputs) {
    // construct output tensor container
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector <uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t) modelDesc_.outputTensors[i].tensorDims[j]);
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

    // statistic inference delay
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *(outputs.get()), dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_infer_cost.push_back(costMs);

    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

/**
 * choose the max probability as the inference result
 * @param outputs reference to model output tensors
 * @param argmax reference to the index of max probability
 * @return status code of whether post-process is successful
 */
APP_ERROR SentimentNet::PostProcess(const std::shared_ptr<std::vector<MxBase::TensorBase>> &outputs,
                                    const std::shared_ptr<std::vector<uint32_t>> &argmax, bool printResult) {
    MxBase::TensorBase &tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }

    // get output tensor info
    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[0];
    uint32_t classNum = outputShape[1];
    void *data = tensor.GetBuffer();

    // get inference result
    for (uint32_t i = 0; i < length; i++) {
        std::vector<float> result = {};
        std::vector<float> softmax_result = {};
        float sum_softmax = 0.0;
        std::string inferOutputTensor = "infer output tensor: [";
        for (uint32_t j = 0; j < classNum; j++) {
            float value = *(reinterpret_cast<float *>(data) + i * classNum + j);
            if (j == classNum - 1) {
                inferOutputTensor = inferOutputTensor + std::to_string(value) + "]";
            } else {
                inferOutputTensor = inferOutputTensor + std::to_string(value) + ",";
            }
            softmax_result.push_back(std::exp(value));
            sum_softmax += softmax_result[j];
        }
        // softmax
        std::string softMaxResult = "softmax result: [";
        for (uint32_t j = 0; j < classNum; j++) {
            float value = softmax_result[j] / sum_softmax;
            if (j == classNum - 1) {
                softMaxResult = softMaxResult + std::to_string(value) + "]";
            } else {
                softMaxResult = softMaxResult + std::to_string(value) + ",";
            }
            result.push_back(value);
        }
        // argmax and get the classification id
        std::vector<float>::iterator maxElement = std::max_element(std::begin(result), std::end(result));
        uint32_t argmaxIndex = maxElement - std::begin(result);
        argmax->push_back(argmaxIndex);
        std::string infer_result = argmaxIndex == 1 ? "1-pos" : "0-neg";

        if (printResult) {
            LogDebug << inferOutputTensor;
            LogDebug << softMaxResult;
            LogDebug << "infer result: " << infer_result;
        }
    }
    return APP_ERR_OK;
}

/**
 * count true positive, false positive, true negative, false negative, calculate real-time accuracy
 * @param labels const reference to the corresponding real labels of input sentences
 * @param startIndex the start index of real labels
 * @param argmax const reference to the model inference result
 */
void SentimentNet::CountPredictResult(const std::vector<uint32_t> &labels, uint32_t startIndex,
                                      const std::vector<uint32_t> &argmax) {
    uint32_t dataSize = argmax.size();

    // compare with ground truth
    for (uint32_t i = 0; i < dataSize; i++) {
        bool positive = false;
        if (labels[i + startIndex] == 1) {
            positive = true;
        }

        if (labels[i + startIndex] == argmax[i]) {
            if (positive) {
                g_true_positive += 1;
            } else {
                g_true_negative += 1;
            }
        } else {
            if (positive) {
                g_false_positive += 1;
            } else {
                g_false_negative += 1;
            }
        }
    }

    uint32_t total = g_true_positive + g_false_positive + g_true_negative + g_false_negative;
    LogInfo << "TP: " << g_true_positive << ", FP: " << g_false_positive
            << ", TN: " << g_true_negative << ", FN: " << g_false_negative;
    LogInfo << "current accuracy: "
            << (g_true_positive + g_true_negative) * 1.0 / total;
}

/**
 * write model inference result to file
 * @param fileName result file name
 * @param argmax const reference of model inference result
 * @return status code of whether writing file is successful
 */
APP_ERROR SentimentNet::WriteResult(const std::string &fileName, const std::vector<uint32_t> &argmax, bool firstInput) {
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
    std::ofstream resultFile(resultPathName, firstInput ? std::ofstream::ate : std::ofstream::app);
    if (resultFile.fail()) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // write inference result into file
    resultFile << "file name is: " << fileName << " review num: " << std::to_string(argmax.size()) << std::endl;
    for (uint32_t i = 0; i < argmax.size(); i++) {
        std::string prediction = argmax.at(i) == 1 ? "positive" : "negative";
        resultFile << std::to_string(i + 1) << "-th review: " << prediction << std::endl;
    }
    resultFile.close();
    return APP_ERR_OK;
}

/**
 * Emotional classification of the input sentences, result is positive or negative
 * @param inferPath const reference to input sentences dir
 * @param fileName const reference to sentences file name
 * @param eval whether do evaluation
 * @param labels const reference to the corresponding real label of input sentences
 * @param startIndex the start index of real labels in curr inference round
 * @return status code of whether the workflow is successful
 */
APP_ERROR SentimentNet::Process(const std::string &inferPath, const std::string &fileName, bool firstInput,
                                bool eval, const std::vector<uint32_t> &labels, const uint32_t startIndex) {
    // read word vector of sentences
    std::shared_ptr<std::vector<MxBase::TensorBase>> inputs = std::make_shared<std::vector<MxBase::TensorBase>>();
    std::string inputSentencesFile = inferPath + fileName;
    APP_ERROR ret = ReadInputTensor(inputSentencesFile, 0, inputs);
    if (ret != APP_ERR_OK) {
        LogError << "Read input ids failed, ret=" << ret << ".";
        return ret;
    }

    // model inference
    std::shared_ptr<std::vector<MxBase::TensorBase>> outputs = std::make_shared<std::vector<MxBase::TensorBase>>();
    ret = Inference(*(inputs.get()), outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }

    // softmax to get final inference result
    std::shared_ptr<std::vector<uint32_t>> argmax = std::make_shared<std::vector<uint32_t>>();
    ret = PostProcess(outputs, argmax, false);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    // save inference result
    ret = WriteResult(fileName, *(argmax.get()), firstInput);
    if (ret != APP_ERR_OK) {
        LogError << "save result failed, ret=" << ret << ".";
        return ret;
    }

    // evaluation
    if (eval) {
        CountPredictResult(labels, startIndex, *(argmax.get()));
    }

    return APP_ERR_OK;
}
