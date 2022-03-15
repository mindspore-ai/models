/*
 *  Copyright 2022 Huawei Technologies Co., Ltd
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

#include "Hourglass.h"
#include <dirent.h>
#include <sys/stat.h>
#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <boost/property_tree/json_parser.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
namespace {
    const int NPOINTS = 16;
    float det[NPOINTS][64][64] = {};
    float ans[NPOINTS][3] = {};
    double total_time = 0;
    std::vector<cv::Mat> gts;
    std::vector<cv::Mat> preds;
    std::vector<float> normalizing;
    const std::vector<std::string> joint_map {
        "ankle", "knee", "hip", "hip", "knee", "ankle", "pelvis", "thorax",
        "neck", "head", "wrist", "elbow", "shoulder", "shoulder", "elbow", "wrist"
    };
    std::unordered_map<std::string, std::unordered_map<std::string, int>> mpii_template {
        {"not visible", {{"shoulder", 0}, {"elbow", 0}, {"wrist", 0}, {"head", 0}, {"neck", 0},
            {"thorax", 0}, {"pelvis", 0}, {"hip", 0}, {"knee", 0}, {"ankle", 0}, {"total", 0}}
        },
        {"visible", {{"shoulder", 0}, {"elbow", 0}, {"wrist", 0}, {"head", 0}, {"neck", 0},
            {"thorax", 0}, {"pelvis", 0}, {"hip", 0}, {"knee", 0}, {"ankle", 0}, {"total", 0}}
        },
        {"all", {{"shoulder", 0}, {"elbow", 0}, {"wrist", 0}, {"head", 0}, {"neck", 0},
            {"thorax", 0}, {"pelvis", 0}, {"hip", 0}, {"knee", 0}, {"ankle", 0}, {"total", 0}}
        }
    };
}  // namespace

void PrintTensorShape(const std::vector<MxBase::TensorDesc> &tensorDescVec, const std::string &tensorName) {
    LogInfo << "The shape of " << tensorName << " is as follows:";
    for (size_t i = 0; i < tensorDescVec.size(); ++i) {
        LogInfo << "  Tensor " << i << ":";
        for (size_t j = 0; j < tensorDescVec[i].tensorDims.size(); ++j) {
            LogInfo << "   dim: " << j << ": " << tensorDescVec[i].tensorDims[j];
        }
    }
}

APP_ERROR Hourglass::Init(const InitParam &initParam) {
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
  return APP_ERR_OK;
}

APP_ERROR Hourglass::DeInit() {
  // dvppWrapper_->DeInit();
  model_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

APP_ERROR Hourglass::ReadInputTensor(const std::string &fileName, MxBase::TensorBase &tensorBase) {
    uint32_t dataSize = 196608;
    float *metaFeatureData = new float[dataSize];
    std::ifstream infile;
    // cppcheck-suppress memleak
    infile.open(fileName, std::ios_base::in | std::ios_base::binary);
    // check data file validity
    infile.read(reinterpret_cast<char*>(metaFeatureData), sizeof(float) * dataSize);
    infile.close();

    MxBase::MemoryData memoryDataDst(dataSize * 4, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(metaFeatureData),
    dataSize * 4, MxBase::MemoryData::MEMORY_HOST_MALLOC);

    auto ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {1, 256, 256, 3};
    tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
    return APP_ERR_OK;
}

APP_ERROR Hourglass::Process(const std::string &inferPath, const std::string &fileName) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};

    MxBase::TensorBase tensorBase0;
    std::string input_file_0 = inferPath + "/00_data/" + fileName;
    APP_ERROR ret = ReadInputTensor(input_file_0, tensorBase0);
    if (ret != APP_ERR_OK) {
        LogError << "Read input file failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase0);
    MxBase::TensorBase tensorBase1;
    std::string input_file_1 = inferPath + "/11_data/" + fileName;
    ret = ReadInputTensor(input_file_1, tensorBase1);
    if (ret != APP_ERR_OK) {
        LogError << "Read input file failed, ret=" << ret << ".";
        return ret;
    }
    inputs.push_back(tensorBase1);

    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    total_time = inferCostTimeMilliSec;
    std::string anns_file = inferPath + "/22_data/" + fileName;
    std::string c_file = inferPath + "/33_data/" + fileName;
    std::string s_file = inferPath + "/44_data/" + fileName;
    std::string n_file = inferPath + "/55_data/" + fileName;
    std::string mat_file = inferPath + "/66_data/" + fileName;
    float *c = readCFromFile(c_file);
    float *s = readSFromFile(s_file);
    float *n = readNFromFile(n_file);
    cv::Mat mat = readMatFromFile(mat_file);
    cv::Mat gt = readGtFromFile(anns_file);
    cv::Mat cropped_preds(NPOINTS, 3, CV_32FC1);
    ret = ParseTensor(outputs[0], outputs[1]);
    if (ret != APP_ERR_OK) {
        LogError << "parse failed, ret=" << ret << ".";
        return ret;
    }
    for (int i = 0; i < cropped_preds.rows; i++) {
        for (int j = 0; j < 2; j++) {
             cropped_preds.at<float>(i, j) = ans[i][j] * 4;
        }
        cropped_preds.at<float>(i, 2) = 1;
    }
    cv::Mat pred = transform_preds(cropped_preds, mat, c, s[0]);
    gts.push_back(gt);
    preds.push_back(pred);
    normalizing.push_back(n[0]);
    return APP_ERR_OK;
}

APP_ERROR Hourglass::Inference(const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> &outputs) {
  auto dtypes = model_->GetOutputDataType();
  for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
      shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
    }
    MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                      deviceId_);
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
  auto startTime = std::chrono::high_resolution_clock::now();
  APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
  auto endTime = std::chrono::high_resolution_clock::now();
  double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();  // save time
  inferCostTimeMilliSec += costMs;
  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR Hourglass::WriteResult(const std::string &imageFile, std::vector<MxBase::TensorBase> &outputs) {
    std::string homePath = "./result_Files";
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        APP_ERROR ret = outputs[i].ToHost();
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "tohost fail.";
            return ret;
        }
        void *netOutput = outputs[i].GetBuffer();
        outputSize = outputs[i].GetByteSize();
        int pos = imageFile.rfind('/');
        std::string fileName(imageFile, pos + 1);
        fileName.replace(fileName.rfind('.'), fileName.size() - fileName.rfind('.'), '_' + std::to_string(i) + ".bin");
        std::string outFileName = homePath + "/" + fileName;
        FILE *outputFile = fopen(outFileName.c_str(), "wb");
        if (outputFile == nullptr) {
            std::cout << "read fail" << std::endl;
            return ret;
        }
        fwrite(netOutput, outputSize, sizeof(char), outputFile);
        fclose(outputFile);
        outputFile = nullptr;
    }
    return APP_ERR_OK;
}

APP_ERROR Hourglass::ParseTensor(MxBase::TensorBase &tensors, MxBase::TensorBase &tensors1) {
    APP_ERROR ret = tensors.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    void* data = tensors.GetBuffer();
    for (int i = 0; i < NPOINTS; i++) {
        for (int j = 0; j < 64; j++) {
            for (int k = 0; k < 64; k++) {
                float x = *(reinterpret_cast<float*>(data) + i*64*64 + j*64 + k);
                det[i][j][k] = x;
            }
        }
    }
    ret = tensors1.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    void* data1 = tensors1.GetBuffer();
    for (size_t i = 0; i < NPOINTS; i++) {
        for (size_t j = 0; j < 3; j++) {
            float y = *(reinterpret_cast<float*>(data1) + i*3 + j);
            ans[i][j] = y;
        }
    }
    for (size_t i = 0; i < NPOINTS; i++) {
        if (ans[i][2] > 0) {
            float x = ans[i][1];
            float y = ans[i][0];
            int xx = static_cast<int>(x);
            int yy = static_cast<int>(y);
            if (xx > 1 && xx < 63 && yy > 1 && yy < 63) {
                float diff_x = det[i][xx][yy + 1] - det[i][xx][yy - 1];
                float diff_y = det[i][xx + 1][yy] - det[i][xx - 1][yy];
                if (diff_x > 0) {
                    y = y + 0.25;
                }
                if (diff_x < 0) {
                    y = y - 0.25;
                }
                if (diff_y > 0) {
                    x = x + 0.25;
                }
                if (diff_y < 0) {
                    x = x - 0.25;
                }
            }
            ans[i][0] = y + 0.5;
            ans[i][1] = x + 0.5;
        }
    }
    return APP_ERR_OK;
}

float* readCFromFile(std::string path) {
  std::ifstream inFile(path, std::ios::in | std::ios::binary);
  float *c = new float[2];
  inFile.read(reinterpret_cast<char *>(c), sizeof(float)*2);
  inFile.close();
  return c;
}

float* readSFromFile(std::string path) {
  std::ifstream inFile(path, std::ios::in | std::ios::binary);
  float *s = new float[1];
  inFile.read(reinterpret_cast<char *>(s), sizeof(float));
  inFile.close();
  return s;
}

float* readNFromFile(std::string path) {
  std::ifstream inFile(path, std::ios::in | std::ios::binary);
  float *n = new float[1];
  inFile.read(reinterpret_cast<char *>(n), sizeof(float));
  inFile.close();
  return n;
}

cv::Mat readMatFromFile(std::string path) {
  std::ifstream inFile(path, std::ios::in | std::ios::binary);
  cv::Mat im(2, 3, CV_32FC1);
  if (!inFile) {
    std::cout << "error" << std::endl;
    return im;
  }
  for (int r = 0; r < im.rows; r++) {
    inFile.read(reinterpret_cast<char *>(im.ptr<uchar>(r)),
                im.cols * im.elemSize());
  }
  inFile.close();
  return im;
}

cv::Mat readGtFromFile(std::string path) {
  std::ifstream inFile(path, std::ios::in | std::ios::binary);
  cv::Mat im(16, 3, CV_32FC1);
  if (!inFile) {
    std::cout << "error" << std::endl;
    return im;
  }
  for (int r = 0; r < im.rows; r++) {
    inFile.read(reinterpret_cast<char *>(im.ptr<uchar>(r)),
                im.cols * im.elemSize());
  }
  inFile.close();
  return im;
}

cv::Mat transform_preds(const cv::Mat &cropped_preds, cv::Mat &mat, const float c[], const float s) {
    cv::Mat pred(NPOINTS, 3, CV_32FC1);
    float h = 200 * s;
    float res = 256.0;
    cv::Mat t = cv::Mat::zeros(3, 3, CV_32FC1);
    t.at<float>(0, 0) = res / h;
    t.at<float>(1, 1) = res / h;
    t.at<float>(0, 2) = res * (-c[0] / h + 0.5);
    t.at<float>(1, 2) = res * (-c[1] / h + 0.5);
    t.at<float>(2, 2) = 1;
    t = t.inv();
    cv::Mat matT = mat.t();
    cv::Mat temp = cropped_preds * matT;
    for (int i = 0; i < temp.rows; i++) {
        cv::Mat new_pt(3, 1, CV_32FC1);
        new_pt.at<float>(0, 0) = temp.at<float>(i, 0);
        new_pt.at<float>(1, 0) = temp.at<float>(i, 1);
        new_pt.at<float>(2, 0) = 1.0;
        new_pt = t * new_pt;
        pred.at<float>(i, 0) = static_cast<int>(new_pt.at<float>(0, 0));
        pred.at<float>(i, 1) = static_cast<int>(new_pt.at<float>(1, 0));
        pred.at<float>(i, 2) = ans[i][2];
    }
    return pred;
}

void eval() {
    int total_num = 3258;
    int num_train = 300;
    float bound = 0.5;
    int idx = 0;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> correct = mpii_template;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> count = mpii_template;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> correct_train = mpii_template;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> count_train = mpii_template;
    for (int i = 0; i < total_num; i++) {
        float normalize = normalizing[i];
        cv::Mat g = gts[i];
        cv::Mat p = preds[i];
        for (int j = 0; j < NPOINTS; j++) {
            std::string joint = joint_map[j];
            std::string vis = "visible";
            if (static_cast<int>(g.at<float>(j, 0)) == 0) {
                continue;
            }
            if (static_cast<int>(g.at<float>(j, 2)) == 0) {
                vis = "not visible";
            }
            if (idx >= num_train) {
                count["all"]["total"] += 1;
                count["all"][joint] += 1;
                count[vis]["total"] += 1;
                count[vis][joint] += 1;
            } else {
                count_train["all"]["total"] += 1;
                count_train["all"][joint] += 1;
                count_train[vis]["total"] += 1;
                count_train[vis][joint] += 1;
            }
            float gx = g.at<float>(j, 0);
            float gy = g.at<float>(j, 1);
            float px = p.at<float>(j, 0);
            float py = p.at<float>(j, 1);
            float norm = std::sqrt((px-gx)*(px-gx)+(py-gy)*(py-gy));
            float error = norm / normalize;
            if (bound > error) {
                if (idx >= num_train) {
                    correct["all"]["total"] += 1;
                    correct["all"][joint] += 1;
                    correct[vis]["total"] += 1;
                    correct[vis][joint] += 1;
                } else {
                    correct_train["all"]["total"] += 1;
                    correct_train["all"][joint] += 1;
                    correct_train[vis]["total"] += 1;
                    correct_train[vis][joint] += 1;
                }
            }
        }
        idx = idx + 1;
    }
    std::unordered_map<std::string, std::unordered_map<std::string, int>>::iterator it;
    for (it = correct.begin(); it != correct.end(); ++it) {
        std::string k = it->first;
        LogInfo << k << ":";
        std::unordered_map<std::string, int> nnc = it->second;
        std::unordered_map<std::string, int>::iterator ir;
         for (ir = nnc.begin(); ir != nnc.end(); ++ir) {
            std::string key = ir->first;
            float val = static_cast<float>(correct[k][key]) / std::max(count[k][key], 1);
            float tra = static_cast<float>(correct_train[k][key]) / std::max(count_train[k][key], 1);
            LogInfo << "Val PCK @, " << bound << " , " << key << " : " <<
            std::setprecision(3) << val << " , count: " << count[k][key];
            LogInfo << "Tra PCK @, " << bound << " , " << key << " : " <<
            std::setprecision(3) << tra << " , count: " << count_train[k][key];
         }
         LogInfo << std::endl;
    }
    LogInfo << "Infer images sum " << total_num << ", cost total time: " << total_time << " ms.";
    LogInfo << "The throughput: " << total_num * 1000 / total_time << " images/sec.";
}
