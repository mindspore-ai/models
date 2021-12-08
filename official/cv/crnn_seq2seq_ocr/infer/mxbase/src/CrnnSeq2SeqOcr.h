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

#ifndef CRNN_SEQTOSEQ_OCR_H
#define CRNN_SEQTOSEQ_OCR_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
  uint32_t deviceId;
  uint32_t classNum;
  uint32_t objectNum;
  uint32_t blankIndex;
  std::string labelPath;
  bool softmax;
  bool checkTensor;
  bool argmax;
  std::string modelPath;
};

class CrnnSeqToSeqOcr {
 public:
  APP_ERROR Init(const InitParam &initParam);
  APP_ERROR DeInit();
  APP_ERROR ReadAndResize(const std::string &imgPath,
                          MxBase::TensorBase *outputTensor);
  APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
                      std::vector<MxBase::TensorBase> *outputs);
  APP_ERROR PostProcess(std::vector<MxBase::TensorBase> *tensors,
                        std::string *textInfos);
  APP_ERROR Process(const std::string &imgPath, std::string *result);
  // get infer time
  double GetInferCostMilliSec() const { return inferCostTimeMilliSec; }

 private:
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
  MxBase::ModelDesc modelDesc_;
  uint32_t deviceId_ = 0;
  double inferCostTimeMilliSec = 0.0;
  std::vector<std::string> words_;
  int eos_id_ = 2;
  uint32_t eval_batch_size = 1;
  uint32_t decoder_hidden_size = 128;
};
#endif
