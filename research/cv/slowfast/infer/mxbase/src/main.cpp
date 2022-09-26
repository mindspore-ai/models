/*
 * Copyright 2022 Huawei Technologies Co., Ltd.
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

#include <dirent.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "MxBase/Log/Log.h"
#include "SlowFast.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LogWarn << "Please input model_path and data_path.";
    return APP_ERR_OK;
  }

  auto slowfast = std::make_shared<SLOWFAST>(0, argv[2], argv[1]);
  printf("Start running\n");
  slowfast->LoadData();
  int data_size = slowfast->get_max();
  LogInfo << "Total: " << data_size << " iters.";
  int batch_size = slowfast->get_batch_size();
  int count = 0;
  std::vector<std::vector<cv::Mat>> slow_pathway;
  std::vector<std::vector<cv::Mat>> fast_pathway;
  std::vector<std::vector<std::vector<float>>> boxes;
  for (int ii = 0; ii < data_size; ii++) {
    count++;
    slowfast->GetData(ii);
    slow_pathway.push_back(slowfast->get_slow_pathway());
    fast_pathway.push_back(slowfast->get_fast_pathway());
    boxes.push_back(slowfast->get_padded_boxes());
    if (count == batch_size) {
      std::vector<float> preds;
      APP_ERROR ret =
          slowfast->Process(slow_pathway, fast_pathway, boxes, &preds);
      LogInfo << "iter:" << ii << "/" << data_size;
      if (ret != APP_ERR_OK) {
        LogError << "slowfast process failed, ret=" << ret << ".";
        return ret;
      }

      slow_pathway.resize(0);
      fast_pathway.resize(0);
      boxes.resize(0);
      count = 0;
    }
  }
  slowfast->post_process();

  return APP_ERR_OK;
}
