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
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "CrnnSeq2SeqOcr.h"
#include "MxBase/Log/Log.h"

namespace CONST_CONFIG {
const uint32_t CLASS_NUM = 37;
const uint32_t OBJECT_NUM = 24;
const uint32_t BLANK_INDEX = 36;
}

APP_ERROR ScanImages(const std::string &path,
                     std::vector<std::string> *imgFiles) {
  DIR *dirPtr = opendir(path.c_str());
  if (dirPtr == nullptr) {
    LogError << "opendir failed. dir: " << path;
    return APP_ERR_INTERNAL_ERROR;
  }
  dirent *direntPtr = nullptr;
  while ((direntPtr = readdir(dirPtr)) != nullptr) {
    std::string fileName = direntPtr->d_name;
    if (fileName == "." || fileName == "..") {
      continue;
    }
    imgFiles->emplace_back(path + "/" + fileName);
  }
  closedir(dirPtr);
  return APP_ERR_OK;
}

void SetInitParam(InitParam *initParam) {
  initParam->modelPath = "../data/model/crnn-seq2seq-ocr-out.om";
}

int main(int argc, char *argv[]) {
  std::string imgPath = "../data/fsns/test/";
  std::ofstream infer_ret("temp_infer_result.txt");
  InitParam initParam = {};
  SetInitParam(&initParam);

  auto crnn = std::make_shared<CrnnSeqToSeqOcr>();
  APP_ERROR ret_init = crnn->Init(initParam);
  if (ret_init != APP_ERR_OK) {
    LogError << "crnn-seq2seq-ocr init failed, ret=" << ret_init << ".";
    return ret_init;
  }

  std::vector<std::string> imgFilePaths;
  ScanImages(imgPath, &imgFilePaths);
  auto startTime = std::chrono::high_resolution_clock::now();

  for (auto &imgFile : imgFilePaths) {
    std::string result = "";
    APP_ERROR ret = crnn->Process(imgFile, &result);
    int nPos = imgFile.find("//") + 2;
    std::string fileName = imgFile.substr(nPos, -1);
    infer_ret << fileName << result << std::endl;
    if (ret != APP_ERR_OK) {
      LogError << "crnn-seq2seq-ocr process failed, ret=" << ret << ".";
      crnn->DeInit();
      return ret;
    }
  }

  crnn->DeInit();
  auto endTime = std::chrono::high_resolution_clock::now();
  infer_ret.close();
  double costMilliSecs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
  double fps = 1000.0 * imgFilePaths.size() / crnn->GetInferCostMilliSec();
  LogInfo << "Run successfully, please run postprocess next." << std::endl;
  LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps
          << " imgs/sec";
}
