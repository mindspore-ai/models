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
#include "MxBase/Log/Log.h"
#include "SiamFCBase.h"
extern std::vector<std::string> all_videos;
namespace {
const char seq_root_path[] = "../../../OTB2013";
const char code_path[] = "../../src";
const char model1_path[] = "../../../convert/om_data_1.om";
const char model2_path[] = "../../../convert/om_data_2.om";
uint32_t deviceId_ = 0;
}  // namespace
int main(int argc, const char** argv) {
  size_t size_v = all_videos.size();
  int jogging_count = 1;
  SiamFCBase siamfc;
  initParam init;
  init.modelPath1 = model1_path;
  init.modelPath2 = model2_path;
  init.deviceId = deviceId_;
  siamfc.Init(init);
  for (size_t i = 0; i < size_v; ++i) {
    param config;
    config.temp_video = all_videos[i];

    APP_ERROR ret = siamfc.GetPath(config, config.temp_video, jogging_count,
                                   seq_root_path, code_path);

    if (ret != APP_ERR_OK) {
      LogError << "Getpath failed , ret =" << ret << ".";
      return 0;
    }
    ret = siamfc.Process(config);
    if (ret != APP_ERR_OK) {
      LogError << "SiamFC process failed,ret= " << ret << ".";
      siamfc.DeInit();
      return 0;
    }
    if (all_videos[i] == "Jogging" && jogging_count == 1) {
      i--;
      jogging_count++;
    }
  }
  siamfc.DeInit();
  return 0;
}
