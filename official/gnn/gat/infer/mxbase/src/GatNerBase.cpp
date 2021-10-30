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

#include "GatNerBase.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <map>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR GatNerBase::LoadLabels(const std::string &labelPath,
                                 std::vector<std::string> *labelMap) {
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

APP_ERROR GatNerBase::Init(const InitParam &initParam) {
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
  ret = LoadLabels(initParam.labelPath, &labelMap_);
  if (ret != APP_ERR_OK) {
    LogError << "Failed to load labels, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR GatNerBase::DeInit() {
  dvppWrapper_->DeInit();
  model_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

APP_ERROR GatNerBase::ReadTensorFromFile(const std::string &file, float *data,
                                         uint32_t size) {
  if (data == NULL) {
    LogError << "input data is invalid.";
    return APP_ERR_COMM_INVALID_POINTER;
  }

  std::ifstream fp(file);
  std::string line;
  while (std::getline(fp, line)) {
    std::string number;
    std::istringstream readstr(line);

    for (uint32_t j = 0; j < size; j++) {
      std::getline(readstr, number, ' ');
      data[j] = atof(number.c_str());
    }
  }

  return APP_ERR_OK;
}

APP_ERROR GatNerBase::ReadTensorFromFile(const std::string &file, int *data,
                                         uint32_t size) {
  if (data == NULL) {
    LogError << "input data is invalid.";
    return APP_ERR_COMM_INVALID_POINTER;
  }

  std::ifstream fp(file);
  std::string line;
  while (std::getline(fp, line)) {
    std::string number;
    std::istringstream readstr(line);
    for (uint32_t j = 0; j < size; j++) {
      std::getline(readstr, number, ' ');
      data[j] = atoi(number.c_str());
    }
  }

  return APP_ERR_OK;
}

APP_ERROR GatNerBase::ReadInputTensor(const std::string &fileName,
                                      uint32_t index,
                                      std::vector<MxBase::TensorBase> *inputs,
                                      const uint32_t size) {
  LogInfo << size;
  float *data = new float[size];
  APP_ERROR ret = ReadTensorFromFile(fileName, data, size);
  if (ret != APP_ERR_OK) {
    LogError << "ReadTensorFromFile failed.";
    return ret;
  }
  const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
  MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE,
                                   deviceId_);
  MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(data), dataSize,
                                   MxBase::MemoryData::MEMORY_HOST_MALLOC);
  ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc and copy failed.";
    return ret;
  }
  std::vector<uint32_t> shape = {1, size};
  inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape,
                                       MxBase::TENSOR_DTYPE_FLOAT32));
  delete[] data;
  return APP_ERR_OK;
}

APP_ERROR GatNerBase::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                std::vector<MxBase::TensorBase> *outputs) {
  auto dtypes = model_->GetOutputDataType();
  for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
      shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
    }
    MxBase::TensorBase tensor(shape, dtypes[i],
                              MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                              deviceId_);
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
  double costMs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
  g_inferCost.push_back(costMs);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR GatNerBase::PostProcess(std::vector<MxBase::TensorBase> *outputs,
                                  std::vector<uint32_t> *argmax) {
  MxBase::TensorBase &tensor = outputs->at(0);
  APP_ERROR ret = tensor.ToHost();
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Tensor deploy to host failed.";
    return ret;
  }
  // check tensor is available
  auto outputShape = tensor.GetShape();
  uint32_t length = outputShape[1];
  uint32_t classNum = outputShape[2];
  LogInfo << "output shape is: " << outputShape[1] << " " << outputShape[2]
          << std::endl;

  void *data = tensor.GetBuffer();
  for (uint32_t i = 0; i < length; i++) {
    std::vector<float> result = {};
    for (uint32_t j = 0; j < classNum; j++) {
      float value = half_float::half_cast<float>(
          *(reinterpret_cast<half_float::half *>(data) + i * classNum + j));
      result.push_back(value);
    }
    // argmax and get the class id
    std::vector<float>::iterator maxElement =
        std::max_element(std::begin(result), std::end(result));
    uint32_t argmaxIndex = maxElement - std::begin(result);
    argmax->push_back(argmaxIndex);
  }

  return APP_ERR_OK;
}

APP_ERROR GatNerBase::CountPredictResult(const std::string &labelFile,
                                         const std::vector<uint32_t> &argmax) {
  int *onehots = new int[NODES * CLASS_NUM];
  APP_ERROR ret = ReadTensorFromFile(labelFile, onehots, NODES * CLASS_NUM);
  if (ret != APP_ERR_OK) {
    LogError << "ReadTensorFromFile failed.";
    return ret;
  }

  uint32_t *data = new uint32_t[NODES];
  for (uint32_t i = 0; i < NODES; i++) {
    for (uint32_t j = 0; j < CLASS_NUM; j++) {
      if (onehots[i * CLASS_NUM + j] == 1) {
        data[i] = j;
        break;
      }
    }
  }

  for (uint32_t i = 0; i < NODES; i++) {
    if (data[i] == argmax[i]) {
      g_TP += 1;
    } else {
      g_FP += 1;
    }
  }
  LogInfo << "TP: " << g_TP << ", FP: " << g_FP;
  delete[] onehots;
  delete[] data;
  return APP_ERR_OK;
}

void GatNerBase::GetClunerLabel(
    const std::vector<uint32_t> &argmax,
    std::multimap<std::string, uint32_t> *clunerMap) {
  for (uint32_t i = 0; i < argmax.size(); i++) {
    clunerMap->insert(
        std::pair<std::string, uint32_t>(labelMap_[argmax[i]], i));
  }
}

APP_ERROR GatNerBase::WriteResult(const std::string &fileName,
                                  const std::vector<uint32_t> &argmax) {
  std::string resultPathName = "result";
  // create result directory when it does not exit
  if (access(resultPathName.c_str(), 0) != 0) {
    int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    if (ret != 0) {
      LogError << "Failed to create result directory: " << resultPathName
               << ", ret = " << ret;
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
  LogInfo << "==============================================================";
  LogInfo << "infer result of " << fileName << " is: ";
  tfile << "file name is: " << fileName << std::endl;

  std::multimap<std::string, uint32_t> clunerMap;
  GetClunerLabel(argmax, &clunerMap);

  for (auto &item : clunerMap) {
    tfile << "node " << item.second << " : " << item.first << std::endl;
  }
  LogInfo << "==============================================================";
  tfile.close();
  return APP_ERR_OK;
}

APP_ERROR GatNerBase::Process(const std::string &inferPath,
                              const std::string &fileName, bool eval) {
  std::vector<MxBase::TensorBase> inputs = {};

  std::string inputFeatureFile = inferPath + "feature.txt";
  APP_ERROR ret = ReadInputTensor(inputFeatureFile, INPUT_FEATURE, &inputs,
                        NODES * FEATURES);
  if (ret != APP_ERR_OK) {
    LogError << "Read input feature file failed, ret=" << ret << ".";
    return ret;
  }

  std::string inputAdjacencyFile = inferPath + "adjacency.txt";
  ret = ReadInputTensor(inputAdjacencyFile, INPUT_ADJACENCY, &inputs,
                                  NODES * NODES);
  if (ret != APP_ERR_OK) {
    LogError << "Read input adjacency failed, ret=" << ret << ".";
    return ret;
  }

  std::vector<MxBase::TensorBase> outputs = {};
  ret = Inference(inputs, &outputs);
  if (ret != APP_ERR_OK) {
    LogError << "Inference failed, ret=" << ret << ".";
    return ret;
  }

  std::vector<uint32_t> argmax;
  ret = PostProcess(&outputs, &argmax);
  if (ret != APP_ERR_OK) {
    LogError << "PostProcess failed, ret=" << ret << ".";
    return ret;
  }

  ret = WriteResult(fileName, argmax);
  if (ret != APP_ERR_OK) {
    LogError << "save result failed, ret=" << ret << ".";
    return ret;
  }

  if (eval) {
    std::string labelFile = inferPath + "label_onehot.txt";
    ret = CountPredictResult(labelFile, argmax);
    if (ret != APP_ERR_OK) {
      LogError << "CalcF1Score read label failed, ret=" << ret << ".";
      return ret;
    }
  }

  return APP_ERR_OK;
}
