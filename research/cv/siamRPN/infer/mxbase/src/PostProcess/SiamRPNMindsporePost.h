/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef SiamRPN_MINSPORE_PORT_H
#define SiamRPN_MINSPORE_PORT_H
#include <algorithm>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include "MxBase/CV/Core/DataType.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace MxBase {
cv::Mat readMatFromFile(std::string path, int height, int width);

class Config {
 public:
  cv::Mat anchors = readMatFromFile("./src/anchors.bin", 1445, 4);
  cv::Mat windows = readMatFromFile("./src/windows.bin", 1445, 1);
  float min_scale = 0.1;
  float max_scale = 10;
  float window_influence = 0.40;
  float penalty_k = 0.22;
  float lr_box = 0.3;
};
class Tracker {
 public:
  int resize_template = 127;
  int resize_detection = 255;
  float context_amount = 0.5;
  // initial var
  float gt_bbox[2500][8];
  std::string path;
  float pred_box[4];
  float scale_x = 0.0;
  float shape[2];
  // read bbox from label
  float bbox[8];
  float box_01[4];
  bool flag = true;
  float target_sz[2];
  float pos[2];
  float origin_target_sz[2];
};

class SiamRPNMindsporePost : public ObjectPostProcessBase {
 public:
  SiamRPNMindsporePost() = default;

  ~SiamRPNMindsporePost() = default;

  SiamRPNMindsporePost(const SiamRPNMindsporePost &other) = default;

  SiamRPNMindsporePost &operator=(const SiamRPNMindsporePost &other);

  APP_ERROR Init(
      const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

  APP_ERROR DeInit() override;

  APP_ERROR Process(const std::vector<TensorBase> &tensors, Config &postConfig,
                    Tracker &track, int idx, int total_num,
                    float result_box[][4], int &template_idx);
  bool IsValidTensors(const std::vector<TensorBase> &tensors) const override;

  void ObjectTrackingOutput(const std::vector<TensorBase> &tensors,
                            Config *postConfig, Tracker &track, int idx,
                            int total_num, float result_box[][4],
                            int &template_idx);

 private:
  void TensorBaseToCVMat(cv::Mat &imageMat, const MxBase::TensorBase &tensor);
  const uint32_t DEFAULT_CLASS_NUM_MS = 80;
  const float DEFAULT_SCORE_THRESH_MS = 0.7;
  const float DEFAULT_IOU_THRESH_MS = 0.5;
  const uint32_t DEFAULT_RPN_MAX_NUM_MS = 1000;
  const uint32_t DEFAULT_MAX_PER_IMG_MS = 128;

  uint32_t classNum_ = DEFAULT_CLASS_NUM_MS;
  float scoreThresh_ = DEFAULT_SCORE_THRESH_MS;
  float iouThresh_ = DEFAULT_IOU_THRESH_MS;
  uint32_t rpnMaxNum_ = DEFAULT_RPN_MAX_NUM_MS;
  uint32_t maxPerImg_ = DEFAULT_MAX_PER_IMG_MS;
};

extern "C" {
std::shared_ptr<MxBase::SiamRPNMindsporePost> GetObjectInstance();
}
}  // namespace MxBase
#endif  // SiamRPN_MINSPORE_PORT_H
